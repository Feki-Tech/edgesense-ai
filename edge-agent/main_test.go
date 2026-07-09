package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestTopicMachineID(t *testing.T) {
	cases := map[string]string{
		"edgesense/sensors/machine-01": "machine-01",
		"edgesense/sensors/a/b":        "b",
		"bare":                         "bare",
	}
	for topic, want := range cases {
		if got := topicMachineID(topic); got != want {
			t.Errorf("topicMachineID(%q) = %q, want %q", topic, got, want)
		}
	}
}

func TestScoreOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]float64
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		for _, k := range []string{"vibration", "temperature", "current"} {
			if _, ok := body[k]; !ok {
				t.Errorf("request missing field %q", k)
			}
		}
		json.NewEncoder(w).Encode(scoreResponse{Score: -0.12, IsAnomaly: true})
	}))
	defer srv.Close()

	hc := &http.Client{Timeout: time.Second}
	sr, err := score(hc, srv.URL, reading{Vibration: 4.2, Temperature: 46, Current: 14.5})
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if !sr.IsAnomaly || sr.Score != -0.12 {
		t.Errorf("unexpected response: %+v", sr)
	}
}

func TestScoreNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	if _, err := score(&http.Client{Timeout: time.Second}, srv.URL, reading{}); err == nil {
		t.Fatal("expected error on 500 response")
	}
}

func TestScoreBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("not json"))
	}))
	defer srv.Close()

	if _, err := score(&http.Client{Timeout: time.Second}, srv.URL, reading{}); err == nil {
		t.Fatal("expected error on malformed body")
	}
}
