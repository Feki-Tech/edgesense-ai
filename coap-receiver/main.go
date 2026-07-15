// EdgeSense CoAP receiver.
//
// Cloud-side bridge for the agent's CoAP uplink (edge-agent, coap://…):
// accepts confirmable POSTs of anomaly events on /events (CBOR, with a JSON
// fallback) and republishes them as canonical JSON to the cloud MQTT broker
// on edgesense/events/<machine_id> — the same topics the MQTT uplink uses —
// so the dashboard, Grafana and demos work identically for both transports.
//
// The 2.04 response is only sent after the broker acknowledged the publish
// (QoS 1 PUBACK); until then the agent's CON retransmits or, on timeout,
// its disk buffer keeps the event. A broker outage behind the receiver is
// answered with 5.03 so the agent buffers — store-and-forward holds
// end-to-end.
//
// Operational state is exposed on EDGESENSE_METRICS_ADDR: Prometheus
// metrics on /metrics, liveness on /healthz.
package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	coapmux "github.com/plgd-dev/go-coap/v3/mux"
	coapnet "github.com/plgd-dev/go-coap/v3/net"
	"github.com/plgd-dev/go-coap/v3/options"
	"github.com/plgd-dev/go-coap/v3/udp"
)

const publishTimeout = 2 * time.Second

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	listen := envOr("EDGESENSE_COAP_LISTEN", ":5683")
	broker := envOr("EDGESENSE_UPLINK_BROKER", "tcp://localhost:12883")
	metricsAddr := envOr("EDGESENSE_METRICS_ADDR", ":8891")

	opts := mqtt.NewClientOptions().
		AddBroker(broker).
		SetClientID("edgesense-coap-receiver").
		SetAutoReconnect(true).
		SetConnectRetry(true).
		SetConnectRetryInterval(2 * time.Second).
		SetMaxReconnectInterval(10 * time.Second).
		SetOrderMatters(false).
		SetOnConnectHandler(func(_ mqtt.Client) {
			brokerUp.Set(1)
			log.Printf("broker connected: %s", broker)
		}).
		SetConnectionLostHandler(func(_ mqtt.Client, err error) {
			brokerUp.Set(0)
			log.Printf("broker connection lost: %v (agents will buffer)", err)
		})
	client := mqtt.NewClient(opts)

	publish := func(topic string, payload []byte) error {
		// Gate on a live connection so the agent gets a fast 5.03 and keeps
		// the event in its own buffer instead of paho's reconnect queue.
		if !client.IsConnectionOpen() {
			return fmt.Errorf("broker not connected")
		}
		tok := client.Publish(topic, 1, false, payload)
		if !tok.WaitTimeout(publishTimeout) {
			return fmt.Errorf("publish timeout on %s", topic)
		}
		return tok.Error()
	}

	serveMetrics(metricsAddr, func() map[string]any {
		return map[string]any{
			"status":           "ok",
			"broker_connected": client.IsConnectionOpen(),
		}
	})
	log.Printf("metrics on %s (/metrics, /healthz)", metricsAddr)

	// ConnectRetry keeps trying in the background; an unreachable broker at
	// startup is survivable — POSTs are answered 5.03 until it comes up.
	client.Connect()

	router := coapmux.NewRouter()
	if err := router.Handle(eventsPath, eventsHandler(publish)); err != nil {
		log.Fatalf("route: %v", err)
	}
	l, err := coapnet.NewListenUDP("udp", listen)
	if err != nil {
		log.Fatalf("listen udp %s: %v", listen, err)
	}
	defer l.Close()
	srv := udp.NewServer(options.WithMux(router))

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(l) }()
	log.Printf("coap receiver on %s (POST %s) → %s", listen, eventsPath, broker)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-sig:
		log.Printf("shutting down")
		srv.Stop()
	case err := <-errCh:
		log.Fatalf("coap server: %v", err)
	}
	client.Disconnect(250)
}
