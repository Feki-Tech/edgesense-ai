// EdgeSense edge agent.
//
// Subscribes to raw sensor readings on MQTT, scores each reading against the
// inference sidecar, and publishes an event to edgesense/events/<machine_id>
// when a reading is anomalous. Only events leave the node.
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
}

type event struct {
	MachineID string  `json:"machine_id"`
	Ts        float64 `json:"ts"`
	Score     float64 `json:"score"`
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
	inferenceURL := envOr("EDGESENSE_INFERENCE_URL", "http://localhost:8800/score")
	sensorTopic := envOr("EDGESENSE_SENSOR_TOPIC", "edgesense/sensors/#")

	httpClient := &http.Client{Timeout: 2 * time.Second}

	opts := mqtt.NewClientOptions().
		AddBroker(broker).
		SetClientID("edgesense-agent").
		SetAutoReconnect(true).
		SetOrderMatters(false)

	client := mqtt.NewClient(opts)
	if tok := client.Connect(); tok.Wait() && tok.Error() != nil {
		log.Fatalf("mqtt connect: %v", tok.Error())
	}
	log.Printf("connected to %s, subscribing %s", broker, sensorTopic)

	handler := func(_ mqtt.Client, msg mqtt.Message) {
		var r reading
		if err := json.Unmarshal(msg.Payload(), &r); err != nil {
			log.Printf("bad payload on %s: %v", msg.Topic(), err)
			return
		}
		if r.MachineID == "" {
			parts := strings.Split(msg.Topic(), "/")
			r.MachineID = parts[len(parts)-1]
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
			Reading:   r,
			AgentTs:   float64(time.Now().UnixNano()) / 1e9,
		}
		payload, _ := json.Marshal(ev)
		topic := fmt.Sprintf("edgesense/events/%s", r.MachineID)
		client.Publish(topic, 0, false, payload)
		log.Printf("ANOMALY %s score=%.4f vib=%.2f temp=%.1f cur=%.2f",
			r.MachineID, sr.Score, r.Vibration, r.Temperature, r.Current)
	}

	if tok := client.Subscribe(sensorTopic, 0, handler); tok.Wait() && tok.Error() != nil {
		log.Fatalf("mqtt subscribe: %v", tok.Error())
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutting down")
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
