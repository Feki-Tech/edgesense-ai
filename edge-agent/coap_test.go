package main

import (
	"encoding/json"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/fxamacker/cbor/v2"
	"github.com/plgd-dev/go-coap/v3/message"
	"github.com/plgd-dev/go-coap/v3/message/codes"
	coapmux "github.com/plgd-dev/go-coap/v3/mux"
	coapnet "github.com/plgd-dev/go-coap/v3/net"
	"github.com/plgd-dev/go-coap/v3/options"
	"github.com/plgd-dev/go-coap/v3/udp"
)

// shortenCoAPTimers makes probe/timeout cycles fast for tests.
func shortenCoAPTimers(t *testing.T) {
	t.Helper()
	oldPost, oldPing, oldProbe := coapPostTimeout, coapPingTimeout, coapProbeInterval
	coapPostTimeout = 2 * time.Second
	coapPingTimeout = 500 * time.Millisecond
	coapProbeInterval = 100 * time.Millisecond
	t.Cleanup(func() {
		coapPostTimeout, coapPingTimeout, coapProbeInterval = oldPost, oldPing, oldProbe
	})
}

// eventSink records events POSTed to an in-process CoAP server.
type eventSink struct {
	mu     sync.Mutex
	events []event
	cf     []message.MediaType
	reply  codes.Code
}

func (s *eventSink) handle(w coapmux.ResponseWriter, r *coapmux.Message) {
	body, err := r.ReadBody()
	if err != nil {
		_ = w.SetResponse(codes.InternalServerError, message.TextPlain, nil)
		return
	}
	cf, _ := r.ContentFormat()
	var ev event
	if err := cbor.Unmarshal(body, &ev); err != nil {
		_ = w.SetResponse(codes.BadRequest, message.TextPlain, nil)
		return
	}
	s.mu.Lock()
	reply := s.reply
	if reply == 0 {
		reply = codes.Changed
	}
	if reply>>5 == 2 {
		s.events = append(s.events, ev)
		s.cf = append(s.cf, cf)
	}
	s.mu.Unlock()
	_ = w.SetResponse(reply, message.TextPlain, nil)
}

func (s *eventSink) setReply(c codes.Code) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reply = c
}

func (s *eventSink) snapshot() []event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]event(nil), s.events...)
}

// startCoAPTestServer runs an in-process go-coap UDP server; addr may be
// "127.0.0.1:0" for a random port. Returns the bound address and a stopper.
func startCoAPTestServer(t *testing.T, addr string, sink *eventSink) (string, func()) {
	t.Helper()
	router := coapmux.NewRouter()
	if err := router.Handle(coapEventsPath, coapmux.HandlerFunc(sink.handle)); err != nil {
		t.Fatalf("route: %v", err)
	}
	l, err := coapnet.NewListenUDP("udp", addr)
	if err != nil {
		t.Fatalf("listen udp %s: %v", addr, err)
	}
	srv := udp.NewServer(options.WithMux(router))
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = srv.Serve(l)
	}()
	bound := l.LocalAddr().String()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			srv.Stop()
			_ = l.Close()
			<-done
		})
	}
	t.Cleanup(stop)
	return bound, stop
}

func newTestCoAPUplink(t *testing.T, addr string, onUp func()) *coapUplink {
	t.Helper()
	u, err := url.Parse("coap://" + addr)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	c, err := newCoAPUplink(u, onUp)
	if err != nil {
		t.Fatalf("newCoAPUplink: %v", err)
	}
	t.Cleanup(c.Close)
	return c
}

func waitUntil(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestNewUplinkTransportSchemeDispatch(t *testing.T) {
	tr, err := newUplinkTransport("coap://receiver.example:5683", nil)
	if err != nil {
		t.Fatalf("coap dispatch: %v", err)
	}
	cu, ok := tr.(*coapUplink)
	if !ok {
		t.Fatalf("coap:// gave %T, want *coapUplink", tr)
	}
	if cu.addr != "receiver.example:5683" {
		t.Errorf("addr = %q", cu.addr)
	}

	tr, err = newUplinkTransport("coap://receiver.example", nil)
	if err != nil {
		t.Fatalf("coap default port: %v", err)
	}
	if got := tr.(*coapUplink).addr; got != "receiver.example:5683" {
		t.Errorf("default port addr = %q, want receiver.example:5683", got)
	}

	tr, err = newUplinkTransport("tcp://localhost:1883", nil)
	if err != nil {
		t.Fatalf("tcp dispatch: %v", err)
	}
	if _, ok := tr.(*mqttUplink); !ok {
		t.Fatalf("tcp:// gave %T, want *mqttUplink", tr)
	}

	if _, err := newUplinkTransport("coaps://receiver:5684", nil); err == nil {
		t.Error("coaps:// should be rejected (DTLS unsupported)")
	}
	if _, err := newUplinkTransport("coap://", nil); err == nil {
		t.Error("coap:// without host should be rejected")
	}
}

func TestMQTTUplinkGatesOnConnection(t *testing.T) {
	// A never-connected paho client: the gate must fail fast so events go to
	// the disk buffer instead of paho's internal reconnect queue.
	m := sharedMQTTUplink(mqtt.NewClient(mqtt.NewClientOptions().AddBroker("tcp://127.0.0.1:1")))
	if m.Connected() {
		t.Fatal("disconnected client reported Connected")
	}
	err := m.Publish(ev("m1", 1))
	if err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Fatalf("Publish = %v, want 'uplink not connected'", err)
	}
}

func TestCoAPPublishDeliversCBOR(t *testing.T) {
	shortenCoAPTimers(t)
	sink := &eventSink{}
	addr, _ := startCoAPTestServer(t, "127.0.0.1:0", sink)

	c := newTestCoAPUplink(t, addr, nil)
	c.Start()
	waitUntil(t, 5*time.Second, "uplink up", c.Connected)

	sent := event{
		MachineID: "machine-07",
		Ts:        1721051.25,
		Score:     -0.4375,
		Reason:    "model+limit",
		Reading:   reading{MachineID: "machine-07", Ts: 1721051.25, Vibration: 4.2, Temperature: 81.5, Current: 14.125},
		AgentTs:   1721052.5,
	}
	if err := c.Publish(sent); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	got := sink.snapshot()
	if len(got) != 1 {
		t.Fatalf("server saw %d events, want 1", len(got))
	}
	if got[0] != sent {
		t.Errorf("event round-trip mismatch:\n got %+v\nwant %+v", got[0], sent)
	}
	sink.mu.Lock()
	cf := sink.cf[0]
	sink.mu.Unlock()
	if cf != message.AppCBOR {
		t.Errorf("content format = %v, want application/cbor (60)", cf)
	}

	// CBOR must also stay compatible with the JSON wire names used by MQTT
	// consumers: decode via a JSON re-encode of the CBOR-decoded event.
	j, _ := json.Marshal(got[0])
	if !strings.Contains(string(j), `"machine_id":"machine-07"`) {
		t.Errorf("json field names lost through cbor round-trip: %s", j)
	}
}

func TestCoAPPublishFailsFastWhenDown(t *testing.T) {
	shortenCoAPTimers(t)
	c := newTestCoAPUplink(t, "127.0.0.1:9", nil) // nothing listens on discard
	if c.Connected() {
		t.Fatal("uplink up without a server")
	}
	start := time.Now()
	err := c.Publish(ev("m1", 1))
	if err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Fatalf("Publish = %v, want fail-fast 'not connected'", err)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("fail-fast publish took %v", d)
	}
}

func TestCoAPRejectedResponseKeepsLinkUp(t *testing.T) {
	shortenCoAPTimers(t)
	sink := &eventSink{}
	addr, _ := startCoAPTestServer(t, "127.0.0.1:0", sink)

	c := newTestCoAPUplink(t, addr, nil)
	c.Start()
	waitUntil(t, 5*time.Second, "uplink up", c.Connected)

	// Receiver up but its broker down → 5.03: buffer the event, keep link up.
	sink.setReply(codes.ServiceUnavailable)
	err := c.Publish(ev("m1", 1))
	if err == nil || !strings.Contains(err.Error(), "rejected") {
		t.Fatalf("Publish = %v, want rejection error", err)
	}
	if !c.Connected() {
		t.Error("5.03 must not mark the CoAP hop down")
	}

	sink.setReply(codes.Changed)
	if err := c.Publish(ev("m1", 2)); err != nil {
		t.Fatalf("Publish after recovery: %v", err)
	}
}

// TestCoAPOutageBuffersAndReplays exercises the full store-and-forward cycle
// over CoAP: publish OK → receiver dies → events spool to the disk buffer →
// receiver returns → onUp flush replays them FIFO and duplicate-free.
func TestCoAPOutageBuffersAndReplays(t *testing.T) {
	shortenCoAPTimers(t)
	sink := &eventSink{}
	addr, stop := startCoAPTestServer(t, "127.0.0.1:0", sink)

	buffer := newTestBuffer(t, 100)
	var c *coapUplink
	flushes := make(chan int, 16)
	onUp := func() {
		n, _ := buffer.DrainTo(c.Publish)
		flushes <- n
	}
	c = newTestCoAPUplink(t, addr, onUp)
	c.Start()
	waitUntil(t, 5*time.Second, "uplink up", c.Connected)

	if err := c.Publish(ev("live-1", 1)); err != nil {
		t.Fatalf("baseline publish: %v", err)
	}

	stop() // uplink outage
	waitUntil(t, 5*time.Second, "outage detected", func() bool { return !c.Connected() })

	// The agent loop buffers exactly the events whose publish failed.
	for i, id := range []string{"out-1", "out-2", "out-3"} {
		e := ev(id, float64(10+i))
		if err := c.Publish(e); err == nil {
			t.Fatalf("publish %s succeeded during outage", id)
		} else if berr := buffer.Add(e); berr != nil {
			t.Fatalf("buffer add: %v", berr)
		}
	}
	if buffer.Len() != 3 {
		t.Fatalf("buffer depth = %d, want 3", buffer.Len())
	}

	// Restore the receiver on the same address; the prober must detect it
	// and fire onUp, which drains the buffer.
	_, _ = startCoAPTestServer(t, addr, sink)
	waitUntil(t, 10*time.Second, "buffer drained", func() bool { return buffer.Len() == 0 })
	waitUntil(t, 5*time.Second, "replay arrived", func() bool { return len(sink.snapshot()) >= 4 })
	time.Sleep(300 * time.Millisecond) // catch any duplicate stragglers

	got := sink.snapshot()
	if len(got) != 4 {
		t.Fatalf("server saw %d events, want exactly 4 (no duplicates): %+v", len(got), got)
	}
	for i, want := range []string{"live-1", "out-1", "out-2", "out-3"} {
		if got[i].MachineID != want {
			t.Errorf("event[%d] = %s, want %s (FIFO order)", i, got[i].MachineID, want)
		}
	}
	// onUp fires on the initial connect too (flushing an empty buffer); sum
	// across all flushes must equal the 3 buffered events, exactly once.
	total := 0
	for {
		select {
		case n := <-flushes:
			total += n
			continue
		default:
		}
		break
	}
	if total != 3 {
		t.Errorf("flushes replayed %d events total, want 3", total)
	}
}

func TestCoAPRecoveryFiresOnUpOnce(t *testing.T) {
	shortenCoAPTimers(t)
	sink := &eventSink{}
	addr, _ := startCoAPTestServer(t, "127.0.0.1:0", sink)

	ups := make(chan struct{}, 16)
	c := newTestCoAPUplink(t, addr, func() { ups <- struct{}{} })
	c.Start()
	waitUntil(t, 5*time.Second, "uplink up", c.Connected)

	// Stays up across several probe intervals without re-firing onUp.
	time.Sleep(5 * coapProbeInterval)
	if got := len(ups); got != 1 {
		t.Fatalf("onUp fired %d times while link stayed up, want 1", got)
	}
}
