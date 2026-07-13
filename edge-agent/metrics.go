// Prometheus metrics and the agent's health endpoint.
//
// The agent exposes /metrics (Prometheus text format) and /healthz (JSON)
// on EDGESENSE_METRICS_ADDR (default :8890). Buffer depth and uplink status
// make store-and-forward observable from the outside.
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
	readingsScored = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_readings_scored_total",
		Help: "Sensor readings scored against the model.",
	}, []string{"machine"})

	scoreFailures = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_score_failures_total",
		Help: "Readings that could not be scored (inference unreachable or error).",
	})

	anomalies = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "edgesense_anomalies_total",
		Help: "Readings flagged anomalous.",
	}, []string{"machine", "reason"})

	eventsPublished = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_events_published_total",
		Help: "Anomaly events delivered to the uplink broker (including buffer replays).",
	})

	eventsBuffered = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edgesense_events_buffered_total",
		Help: "Anomaly events written to the store-and-forward buffer.",
	})

	bufferDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "edgesense_buffer_depth",
		Help: "Events currently waiting in the store-and-forward buffer.",
	})

	uplinkUp = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "edgesense_uplink_connected",
		Help: "1 when the uplink broker connection is open.",
	})

	inferenceLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "edgesense_inference_latency_seconds",
		Help:    "Latency of successful inference sidecar calls.",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 12), // 1ms .. ~4s
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
