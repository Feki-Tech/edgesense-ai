// Prometheus metrics and the receiver's health endpoint.
package main

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	eventsReceived = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_coap_events_received_total",
		Help: "CoAP event POSTs received (before decoding).",
	})

	eventsRepublished = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_coap_events_republished_total",
		Help: "Events republished to the cloud MQTT broker (PUBACK received).",
	})

	eventsRejected = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_coap_events_rejected_total",
		Help: "Malformed event POSTs rejected with 4.xx.",
	})

	republishFailures = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_coap_republish_failures_total",
		Help: "Events answered 5.03 because the broker publish failed.",
	})

	brokerUp = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "edgesense_coap_broker_connected",
		Help: "1 while the cloud MQTT broker connection is open.",
	})
)

// metricsMux serves /metrics and /healthz; health supplies the healthz body.
func metricsMux(health func() map[string]any) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(health())
	})
	return mux
}

func serveMetrics(addr string, health func() map[string]any) {
	go func() {
		if err := http.ListenAndServe(addr, metricsMux(health)); err != nil {
			log.Printf("metrics server on %s: %v", addr, err)
		}
	}()
}
