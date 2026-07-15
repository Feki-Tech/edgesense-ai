// CoAP uplink transport for constrained links (LTE-M, NB-IoT, satellite).
//
// Events are sent as confirmable (CON) POSTs to /events on a CoAP-to-MQTT
// receiver (see coap-receiver/). CON retransmission gives at-least-once
// delivery, mirroring MQTT QoS 1. Payloads are CBOR-encoded (RFC 8949) to
// keep them compact on metered links.
//
// CoAP over UDP is connectionless, so "connected" is defined by exchange
// outcomes: the link is up while POSTs succeed, and a lightweight CoAP ping
// probes it whenever there has been no recent successful exchange. A failed
// exchange marks the link down (publishes then fail fast into the disk
// buffer) and the prober re-dials — refreshing DNS — until a ping succeeds,
// which fires onUp to flush the buffer.
package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net"
	"net/url"
	"sync"
	"time"

	"github.com/fxamacker/cbor/v2"
	"github.com/plgd-dev/go-coap/v3/message"
	"github.com/plgd-dev/go-coap/v3/udp"
	udpclient "github.com/plgd-dev/go-coap/v3/udp/client"
)

const coapEventsPath = "/events"

// Variables (not constants) so tests can shorten them.
var (
	coapPostTimeout   = 5 * time.Second // covers the first CON retransmit (2s ACK timeout)
	coapPingTimeout   = 2 * time.Second
	coapProbeInterval = 3 * time.Second
)

type coapUplink struct {
	addr string // host:port of the CoAP receiver
	onUp func()

	mu     sync.Mutex
	conn   *udpclient.Conn
	up     bool
	lastOK time.Time // last successful exchange (POST or ping)

	done chan struct{}
	once sync.Once
}

func newCoAPUplink(u *url.URL, onUp func()) (*coapUplink, error) {
	if u.Hostname() == "" {
		return nil, fmt.Errorf("coap uplink url %q: missing host", u.String())
	}
	port := u.Port()
	if port == "" {
		port = "5683" // IANA default CoAP port
	}
	return &coapUplink{
		addr: net.JoinHostPort(u.Hostname(), port),
		onUp: onUp,
		done: make(chan struct{}),
	}, nil
}

func (c *coapUplink) Publish(ev event) error {
	payload, err := cbor.Marshal(ev)
	if err != nil {
		return err
	}
	// Gate on the probed link state: during an outage every CON POST would
	// otherwise block through its retransmission window. The disk buffer
	// owns offline events; the prober detects recovery and flushes it.
	c.mu.Lock()
	conn, up := c.conn, c.up
	c.mu.Unlock()
	if !up || conn == nil {
		return fmt.Errorf("coap uplink not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), coapPostTimeout)
	defer cancel()
	resp, err := conn.Post(ctx, coapEventsPath, message.AppCBOR, bytes.NewReader(payload))
	if err != nil {
		c.markDown(conn)
		return fmt.Errorf("coap post: %w", err)
	}
	// Any response proves the CoAP hop is alive.
	c.touch()
	if code := resp.Code(); code>>5 != 2 { // CoAP code class 2.xx = success
		// e.g. 5.03 while the receiver's broker is down: the event was not
		// accepted, so the caller buffers it; the link itself stays up and
		// the periodic flush retries until the receiver accepts again.
		return fmt.Errorf("coap post rejected: %v", code)
	}
	return nil
}

func (c *coapUplink) Connected() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.up
}

func (c *coapUplink) Start() {
	go c.probeLoop()
}

func (c *coapUplink) Close() {
	c.once.Do(func() { close(c.done) })
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
	c.up = false
}

func (c *coapUplink) probeLoop() {
	t := time.NewTicker(coapProbeInterval)
	defer t.Stop()
	c.probe() // immediate first probe so startup state settles quickly
	for {
		select {
		case <-c.done:
			return
		case <-t.C:
			c.probe()
		}
	}
}

// probe keeps the link state honest when there is no publish traffic and
// detects recovery after an outage. It is only ever run by probeLoop.
func (c *coapUplink) probe() {
	c.mu.Lock()
	conn, up, lastOK := c.conn, c.up, c.lastOK
	c.mu.Unlock()

	if up && time.Since(lastOK) < coapProbeInterval {
		return // recent successful exchange already proved liveness
	}

	if up && conn != nil {
		if err := c.ping(conn); err != nil {
			log.Printf("coap uplink lost: %v (events will be buffered)", err)
			c.markDown(conn)
		} else {
			c.touch()
		}
		return
	}

	// Down: dial a fresh socket (re-resolving DNS — the receiver may have
	// restarted with a new address) and probe it.
	conn, err := udp.Dial(c.addr)
	if err != nil {
		return // e.g. DNS failure; stay down, retry next tick
	}
	if err := c.ping(conn); err != nil {
		conn.Close()
		return
	}
	c.markUp(conn)
}

func (c *coapUplink) ping(conn *udpclient.Conn) error {
	ctx, cancel := context.WithTimeout(context.Background(), coapPingTimeout)
	defer cancel()
	return conn.Ping(ctx)
}

// touch records a successful exchange on the live connection.
func (c *coapUplink) touch() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastOK = time.Now()
}

// markDown drops the failed connection so the prober re-dials.
func (c *coapUplink) markDown(failed *udpclient.Conn) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn == failed && c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
	c.up = false
}

// markUp installs a healthy connection and fires onUp on a down→up edge.
func (c *coapUplink) markUp(conn *udpclient.Conn) {
	c.mu.Lock()
	if c.conn != nil && c.conn != conn {
		c.conn.Close()
	}
	c.conn = conn
	wasUp := c.up
	c.up = true
	c.lastOK = time.Now()
	c.mu.Unlock()

	if !wasUp {
		log.Printf("coap uplink connected: %s", c.addr)
		if c.onUp != nil {
			go c.onUp()
		}
	}
}
