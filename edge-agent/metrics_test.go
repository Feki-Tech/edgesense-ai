package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHealthzReportsAgentState(t *testing.T) {
	srv := httptest.NewServer(metricsMux(func() map[string]any {
		return map[string]any{"status": "ok", "uplink_connected": true, "buffer_depth": 2}
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz status = %d, want 200", resp.StatusCode)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("healthz JSON: %v", err)
	}
	if body["status"] != "ok" || body["uplink_connected"] != true || body["buffer_depth"] != float64(2) {
		t.Fatalf("unexpected healthz body: %v", body)
	}
}

func TestMetricsEndpointExposesEdgesenseSeries(t *testing.T) {
	// Touch a labelled counter so its series is present in the output.
	readingsScored.WithLabelValues("machine-test").Inc()
	bufferDepth.Set(3)

	srv := httptest.NewServer(metricsMux(func() map[string]any { return nil }))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)

	for _, want := range []string{
		`edgesense_readings_scored_total{machine="machine-test"}`,
		"edgesense_buffer_depth 3",
		"edgesense_uplink_connected",
		"edgesense_events_published_total",
		"edgesense_inference_latency_seconds_bucket",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("/metrics missing %q", want)
		}
	}
}
