package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/fxamacker/cbor/v2"
	"github.com/plgd-dev/go-coap/v3/message"
	"github.com/plgd-dev/go-coap/v3/message/codes"
	coapmux "github.com/plgd-dev/go-coap/v3/mux"
	coapnet "github.com/plgd-dev/go-coap/v3/net"
	"github.com/plgd-dev/go-coap/v3/options"
	"github.com/plgd-dev/go-coap/v3/udp"
)

func sampleEvent(machine string) event {
	return event{
		MachineID: machine,
		Ts:        1700000000.25,
		Score:     0.93,
		Reason:    "vibration spike",
		Reading: reading{
			MachineID:   machine,
			Ts:          1700000000.25,
			Vibration:   4.2,
			Temperature: 71.5,
			Current:     12.1,
		},
		AgentTs: 1700000000.5,
	}
}

// fakeBroker records republished messages; err (when set) simulates a
// broker outage.
type fakeBroker struct {
	mu     sync.Mutex
	topics []string
	bodies [][]byte
	err    error
}

func (f *fakeBroker) publish(topic string, payload []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	f.topics = append(f.topics, topic)
	f.bodies = append(f.bodies, append([]byte(nil), payload...))
	return nil
}

func (f *fakeBroker) setErr(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.err = err
}

func (f *fakeBroker) published() ([]string, [][]byte) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.topics...), append([][]byte(nil), f.bodies...)
}

// startReceiver runs the real handler on an in-process UDP server and
// returns its address.
func startReceiver(t *testing.T, broker *fakeBroker) string {
	t.Helper()
	router := coapmux.NewRouter()
	if err := router.Handle(eventsPath, eventsHandler(broker.publish)); err != nil {
		t.Fatalf("route: %v", err)
	}
	l, err := coapnet.NewListenUDP("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := udp.NewServer(options.WithMux(router))
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = srv.Serve(l)
	}()
	t.Cleanup(func() {
		srv.Stop()
		_ = l.Close()
		<-done
	})
	return l.LocalAddr().String()
}

func post(t *testing.T, addr string, cf message.MediaType, payload []byte) codes.Code {
	t.Helper()
	conn, err := udp.Dial(addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp, err := conn.Post(ctx, eventsPath, cf, bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	return resp.Code()
}

func TestReceiverRepublishesCBOR(t *testing.T) {
	broker := &fakeBroker{}
	addr := startReceiver(t, broker)

	want := sampleEvent("press-17")
	payload, err := cbor.Marshal(want)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if code := post(t, addr, message.AppCBOR, payload); code != codes.Changed {
		t.Fatalf("code = %v, want 2.04 Changed", code)
	}

	topics, bodies := broker.published()
	if len(topics) != 1 || topics[0] != "edgesense/events/press-17" {
		t.Fatalf("topics = %v, want [edgesense/events/press-17]", topics)
	}
	var got event
	if err := json.Unmarshal(bodies[0], &got); err != nil {
		t.Fatalf("republished payload is not JSON: %v", err)
	}
	if got != want {
		t.Fatalf("republished event = %+v, want %+v", got, want)
	}
	// Dashboard-facing payload must use the agent's JSON field names.
	var fields map[string]any
	_ = json.Unmarshal(bodies[0], &fields)
	for _, k := range []string{"machine_id", "ts", "score", "reading", "agent_ts"} {
		if _, ok := fields[k]; !ok {
			t.Fatalf("republished JSON missing %q: %s", k, bodies[0])
		}
	}
}

func TestReceiverAcceptsJSONFallback(t *testing.T) {
	broker := &fakeBroker{}
	addr := startReceiver(t, broker)

	payload, _ := json.Marshal(sampleEvent("m-json"))
	if code := post(t, addr, message.AppJSON, payload); code != codes.Changed {
		t.Fatalf("code = %v, want 2.04 Changed", code)
	}
	topics, _ := broker.published()
	if len(topics) != 1 || topics[0] != "edgesense/events/m-json" {
		t.Fatalf("topics = %v", topics)
	}
}

func TestReceiverRejectsBadPayloads(t *testing.T) {
	broker := &fakeBroker{}
	addr := startReceiver(t, broker)

	noID, _ := cbor.Marshal(sampleEvent(""))
	wildcard, _ := cbor.Marshal(sampleEvent("a/#b"))
	cases := []struct {
		name    string
		cf      message.MediaType
		payload []byte
	}{
		{"garbage", message.AppCBOR, []byte{0xff, 0x00, 0x13}},
		{"missing machine_id", message.AppCBOR, noID},
		{"topic metacharacters", message.AppCBOR, wildcard},
		{"unsupported content format", message.TextPlain, []byte("hello")},
	}
	for _, tc := range cases {
		if code := post(t, addr, tc.cf, tc.payload); code != codes.BadRequest {
			t.Fatalf("%s: code = %v, want 4.00 BadRequest", tc.name, code)
		}
	}
	if topics, _ := broker.published(); len(topics) != 0 {
		t.Fatalf("rejected payloads reached the broker: %v", topics)
	}
}

func TestReceiverReturns503WhenBrokerDown(t *testing.T) {
	broker := &fakeBroker{}
	broker.setErr(fmt.Errorf("broker not connected"))
	addr := startReceiver(t, broker)

	payload, _ := cbor.Marshal(sampleEvent("m1"))
	if code := post(t, addr, message.AppCBOR, payload); code != codes.ServiceUnavailable {
		t.Fatalf("code = %v, want 5.03 ServiceUnavailable", code)
	}

	// Broker recovers → same event goes through (agent-side replay path).
	broker.setErr(nil)
	if code := post(t, addr, message.AppCBOR, payload); code != codes.Changed {
		t.Fatalf("after recovery code = %v, want 2.04 Changed", code)
	}
	if topics, _ := broker.published(); len(topics) != 1 {
		t.Fatalf("published %d times, want 1", len(topics))
	}
}

func TestDecodeEventWithoutContentFormat(t *testing.T) {
	want := sampleEvent("m2")

	asJSON, _ := json.Marshal(want)
	got, err := decodeEvent(asJSON, 0, false)
	if err != nil || got != want {
		t.Fatalf("json sniff: got %+v, err %v", got, err)
	}

	asCBOR, _ := cbor.Marshal(want)
	got, err = decodeEvent(asCBOR, 0, false)
	if err != nil || got != want {
		t.Fatalf("cbor sniff: got %+v, err %v", got, err)
	}
}

func TestValidate(t *testing.T) {
	if err := validate(sampleEvent("ok-1")); err != nil {
		t.Fatalf("valid event rejected: %v", err)
	}
	for _, id := range []string{"", "a/b", "a+#", "nul\x00"} {
		if err := validate(sampleEvent(id)); err == nil {
			t.Fatalf("machine_id %q accepted, want error", id)
		}
	}
}
