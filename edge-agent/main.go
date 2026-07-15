// EdgeSense edge agent.
//
// Subscribes to raw sensor readings on the local MQTT broker, scores each
// reading against the inference sidecar, and publishes an event when a
// reading is anomalous. Only events leave the node. The event topic is
// edgesense/events/<machine_id> in the legacy layout, or
// es/<org>/<site>/<machine_id>/events when tenant namespacing is enabled via
// EDGESENSE_ORG / EDGESENSE_SITE (see topics.go and PLATFORM.md §4.4).
//
// Events leave through a pluggable uplink transport selected by
// EDGESENSE_UPLINK_URL (uplink.go): MQTT by default (EDGESENSE_UPLINK_BROKER,
// e.g. a cloud broker over a flaky LTE link; defaults to the local broker),
// or CoAP/UDP (coap://host:port, see coap.go) for constrained links. Optional
// MQTT credentials: EDGESENSE_UPLINK_USERNAME/_PASSWORD for the uplink,
// EDGESENSE_BROKER_USERNAME/_PASSWORD for the local broker. Events
// that cannot be published (uplink outage) are buffered on disk and flushed
// on reconnect — no event is lost.
//
// Operational state is exposed on EDGESENSE_METRICS_ADDR: Prometheus
// metrics on /metrics, liveness on /healthz (see metrics.go).
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
)

const (
	publishTimeout = 2 * time.Second
	flushInterval  = 30 * time.Second
	statusInterval = 2 * time.Second
	bufferCapacity = 10_000
)

type reading struct {
	MachineID   string  `json:"machine_id"`
	Ts          float64 `json:"ts"`
	Vibration   float64 `json:"vibration"`
	Temperature float64 `json:"temperature"`
	Current     float64 `json:"current"`
}

type scoreResponse struct {
	Score     float64 `json:"score"`
	IsAnomaly bool    `json:"is_anomaly"`
	Reason    string  `json:"reason"`
}

type event struct {
	MachineID string  `json:"machine_id"`
	Ts        float64 `json:"ts"`
	Score     float64 `json:"score"`
	Reason    string  `json:"reason,omitempty"`
	Reading   reading `json:"reading"`
	AgentTs   float64 `json:"agent_ts"`
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	broker := envOr("EDGESENSE_BROKER", "tcp://localhost:11883")
	uplinkURL := envOr("EDGESENSE_UPLINK_URL", envOr("EDGESENSE_UPLINK_BROKER", broker))
	inferenceURL := envOr("EDGESENSE_INFERENCE_URL", "http://localhost:8800/score")
	layout := layoutFromEnv()
	sensorTopic := envOr("EDGESENSE_SENSOR_TOPIC", layout.defaultSensorFilter())
	bufferPath := envOr("EDGESENSE_BUFFER", "event-buffer.jsonl")
	metricsAddr := envOr("EDGESENSE_METRICS_ADDR", ":8890")
	splitUplink := uplinkURL != broker

	localUser, localPass := os.Getenv("EDGESENSE_BROKER_USERNAME"), os.Getenv("EDGESENSE_BROKER_PASSWORD")
	uplinkUser, uplinkPass := os.Getenv("EDGESENSE_UPLINK_USERNAME"), os.Getenv("EDGESENSE_UPLINK_PASSWORD")
	if !splitUplink && localUser == "" {
		// single-broker mode: the one shared client also publishes events,
		// so uplink credentials apply to it
		localUser, localPass = uplinkUser, uplinkPass
	}

	if layout.namespaced {
		log.Printf("topic layout: namespaced es/%s/%s/<machine>/…", layout.org, layout.site)
	}

	httpClient := &http.Client{Timeout: 2 * time.Second}
	buffer := newEventBuffer(bufferPath, bufferCapacity)
	bufferDepth.Set(float64(buffer.Len())) // events may have survived a restart

	var localClient mqtt.Client
	var uplink uplinkTransport

	publishEvent := func(ev event) error {
		if err := uplink.Publish(ev); err != nil {
			return err
		}
		eventsPublished.Inc()
		return nil
	}

	flush := func(trigger string) {
		n, err := buffer.DrainTo(publishEvent)
		bufferDepth.Set(float64(buffer.Len()))
		if n > 0 {
			log.Printf("flushed %d buffered event(s) (%s)", n, trigger)
		}
		if err != nil {
			log.Printf("buffer flush incomplete (%s): %v (%d still pending)",
				trigger, err, buffer.Len())
		}
	}

	handler := func(_ mqtt.Client, msg mqtt.Message) {
		var r reading
		if err := json.Unmarshal(msg.Payload(), &r); err != nil {
			log.Printf("bad payload on %s: %v", msg.Topic(), err)
			return
		}
		if r.MachineID == "" {
			r.MachineID = machineIDFromTopic(msg.Topic())
		}

		sr, err := score(httpClient, inferenceURL, r)
		if err != nil {
			scoreFailures.Inc()
			log.Printf("score failed for %s: %v", r.MachineID, err)
			return
		}
		readingsScored.WithLabelValues(r.MachineID).Inc()
		if !sr.IsAnomaly {
			return
		}
		anomalies.WithLabelValues(r.MachineID, sr.Reason).Inc()

		ev := event{
			MachineID: r.MachineID,
			Ts:        r.Ts,
			Score:     sr.Score,
			Reason:    sr.Reason,
			Reading:   r,
			AgentTs:   float64(time.Now().UnixNano()) / 1e9,
		}
		if err := publishEvent(ev); err != nil {
			if berr := buffer.Add(ev); berr != nil {
				log.Printf("EVENT LOST for %s (publish: %v, buffer: %v)", r.MachineID, err, berr)
				return
			}
			eventsBuffered.Inc()
			bufferDepth.Set(float64(buffer.Len()))
			log.Printf("uplink publish failed (%v), event buffered (%d pending)", err, buffer.Len())
			return
		}
		log.Printf("ANOMALY %s score=%.4f reason=%s vib=%.2f temp=%.1f cur=%.2f",
			r.MachineID, sr.Score, sr.Reason, r.Vibration, r.Temperature, r.Current)
	}

	localOpts := mqtt.NewClientOptions().
		AddBroker(broker).
		SetClientID("edgesense-agent").
		SetAutoReconnect(true).
		SetOrderMatters(false).
		SetUsername(localUser).
		SetPassword(localPass).
		SetOnConnectHandler(func(c mqtt.Client) {
			if tok := c.Subscribe(sensorTopic, 0, handler); tok.Wait() && tok.Error() != nil {
				log.Printf("subscribe failed: %v", tok.Error())
				return
			}
			log.Printf("connected to local broker %s, subscribed %s", broker, sensorTopic)
			if !splitUplink {
				go flush("reconnect")
			}
		}).
		SetConnectionLostHandler(func(_ mqtt.Client, err error) {
			log.Printf("local broker connection lost: %v", err)
		})
	localClient = mqtt.NewClient(localOpts)

	if splitUplink {
		var err error
		uplink, err = newUplinkTransport(uplinkURL, func() { flush("uplink reconnect") },
			uplinkOptions{layout: layout, username: uplinkUser, password: uplinkPass})
		if err != nil {
			log.Fatalf("uplink: %v", err)
		}
	} else {
		uplink = sharedMQTTUplink(localClient, layout)
	}

	serveMetrics(metricsAddr, func() map[string]any {
		return map[string]any{
			"status":           "ok",
			"uplink_connected": uplink.Connected(),
			"buffer_depth":     buffer.Len(),
		}
	})
	log.Printf("metrics on %s (/metrics, /healthz)", metricsAddr)

	if tok := localClient.Connect(); tok.Wait() && tok.Error() != nil {
		log.Fatalf("mqtt connect (local %s): %v", broker, tok.Error())
	}
	if splitUplink {
		// Transports connect/probe in the background; an unreachable uplink
		// at startup is survivable — events buffer until it comes up.
		uplink.Start()
		log.Printf("uplink: %s (store-and-forward active)", uplinkURL)
	}

	ticker := time.NewTicker(flushInterval)
	defer ticker.Stop()
	go func() {
		for range ticker.C {
			if uplink.Connected() && buffer.Len() > 0 {
				flush("periodic")
			}
		}
	}()

	statusTicker := time.NewTicker(statusInterval)
	defer statusTicker.Stop()
	go func() {
		for range statusTicker.C {
			if uplink.Connected() {
				uplinkUp.Set(1)
			} else {
				uplinkUp.Set(0)
			}
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Printf("shutting down (%d buffered events retained)", buffer.Len())
	uplink.Close()
	localClient.Disconnect(250)
}

func score(hc *http.Client, url string, r reading) (*scoreResponse, error) {
	body, _ := json.Marshal(map[string]float64{
		"vibration":   r.Vibration,
		"temperature": r.Temperature,
		"current":     r.Current,
	})
	start := time.Now()
	resp, err := hc.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inference returned %s", resp.Status)
	}
	var sr scoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, err
	}
	inferenceLatency.Observe(time.Since(start).Seconds())
	return &sr, nil
}
