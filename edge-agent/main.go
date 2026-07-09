// EdgeSense edge agent.
//
// Subscribes to raw sensor readings on MQTT, scores each reading against the
// inference sidecar, and publishes an event to edgesense/events/<machine_id>
// when a reading is anomalous. Only events leave the node. Events that cannot
// be published (broker outage) are buffered on disk and flushed on reconnect.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
)

const (
	publishTimeout = 2 * time.Second
	flushInterval  = 30 * time.Second
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

// topicMachineID extracts the machine id from a topic like
// "edgesense/sensors/<machine_id>".
func topicMachineID(topic string) string {
	parts := strings.Split(topic, "/")
	return parts[len(parts)-1]
}

func main() {
	broker := envOr("EDGESENSE_BROKER", "tcp://localhost:11883")
	inferenceURL := envOr("EDGESENSE_INFERENCE_URL", "http://localhost:8800/score")
	sensorTopic := envOr("EDGESENSE_SENSOR_TOPIC", "edgesense/sensors/#")
	bufferPath := envOr("EDGESENSE_BUFFER", "event-buffer.jsonl")

	httpClient := &http.Client{Timeout: 2 * time.Second}
	buffer := newEventBuffer(bufferPath, bufferCapacity)

	var client mqtt.Client

	publishEvent := func(ev event) error {
		payload, err := json.Marshal(ev)
		if err != nil {
			return err
		}
		topic := fmt.Sprintf("edgesense/events/%s", ev.MachineID)
		tok := client.Publish(topic, 1, false, payload)
		if !tok.WaitTimeout(publishTimeout) {
			return fmt.Errorf("publish timeout on %s", topic)
		}
		return tok.Error()
	}

	flush := func(trigger string) {
		n, err := buffer.DrainTo(publishEvent)
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
			r.MachineID = topicMachineID(msg.Topic())
		}

		sr, err := score(httpClient, inferenceURL, r)
		if err != nil {
			log.Printf("score failed for %s: %v", r.MachineID, err)
			return
		}
		if !sr.IsAnomaly {
			return
		}

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
			log.Printf("publish failed (%v), event buffered (%d pending)", err, buffer.Len())
			return
		}
		log.Printf("ANOMALY %s score=%.4f reason=%s vib=%.2f temp=%.1f cur=%.2f",
			r.MachineID, sr.Score, sr.Reason, r.Vibration, r.Temperature, r.Current)
	}

	opts := mqtt.NewClientOptions().
		AddBroker(broker).
		SetClientID("edgesense-agent").
		SetAutoReconnect(true).
		SetOrderMatters(false).
		SetOnConnectHandler(func(c mqtt.Client) {
			if tok := c.Subscribe(sensorTopic, 0, handler); tok.Wait() && tok.Error() != nil {
				log.Printf("subscribe failed: %v", tok.Error())
				return
			}
			log.Printf("connected to %s, subscribed %s", broker, sensorTopic)
			go flush("reconnect")
		}).
		SetConnectionLostHandler(func(_ mqtt.Client, err error) {
			log.Printf("connection lost: %v (events will be buffered)", err)
		})

	client = mqtt.NewClient(opts)
	if tok := client.Connect(); tok.Wait() && tok.Error() != nil {
		log.Fatalf("mqtt connect: %v", tok.Error())
	}

	ticker := time.NewTicker(flushInterval)
	defer ticker.Stop()
	go func() {
		for range ticker.C {
			if client.IsConnected() && buffer.Len() > 0 {
				flush("periodic")
			}
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Printf("shutting down (%d buffered events retained)", buffer.Len())
	client.Disconnect(250)
}

func score(hc *http.Client, url string, r reading) (*scoreResponse, error) {
	body, _ := json.Marshal(map[string]float64{
		"vibration":   r.Vibration,
		"temperature": r.Temperature,
		"current":     r.Current,
	})
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
	return &sr, nil
}
