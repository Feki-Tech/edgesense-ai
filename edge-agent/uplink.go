// Pluggable uplink transports.
//
// The agent publishes anomaly events through an uplinkTransport selected by
// EDGESENSE_UPLINK_URL scheme dispatch: coap://host:port uses CoAP over UDP
// (see coap.go), anything else (tcp://, ssl://, ws://, ...) uses MQTT — the
// historical default. Both transports share the same contract: Publish either
// delivers the event at-least-once or returns an error so the caller can
// spool it to the disk buffer, and Connected gates publishing so the buffer
// stays the single owner of offline events.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
)

// uplinkTransport is how anomaly events leave the node.
type uplinkTransport interface {
	// Publish delivers one event with at-least-once semantics (MQTT QoS 1
	// or CoAP confirmable POST). An error means the event was not accepted
	// upstream and must be buffered by the caller.
	Publish(ev event) error
	// Connected reports whether the uplink is currently believed healthy.
	// Publishes are gated on it so outages fail fast into the buffer.
	Connected() bool
	// Start begins connecting / probing in the background. An unreachable
	// uplink at startup is survivable: events buffer until it comes up.
	Start()
	// Close releases the transport.
	Close()
}

// newUplinkTransport builds the transport for an uplink URL. onUp runs on
// every down→up transition (used to flush the disk buffer).
func newUplinkTransport(rawURL string, onUp func()) (uplinkTransport, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("uplink url %q: %w", rawURL, err)
	}
	switch u.Scheme {
	case "coap":
		return newCoAPUplink(u, onUp)
	case "coaps":
		return nil, fmt.Errorf("uplink url %q: coaps (DTLS) is not supported yet", rawURL)
	default:
		// MQTT owns every other scheme paho accepts (tcp, ssl, ws, wss, ...).
		return newMQTTUplink(rawURL, onUp), nil
	}
}

// mqttUplink publishes events over MQTT with QoS 1. It either owns a
// dedicated client (split uplink) or wraps the shared local-broker client.
type mqttUplink struct {
	client mqtt.Client
	owned  bool // whether Start/Close manage the client lifecycle
}

// newMQTTUplink creates an uplink with its own MQTT client that keeps
// retrying in the background once started.
func newMQTTUplink(brokerURL string, onUp func()) *mqttUplink {
	opts := mqtt.NewClientOptions().
		AddBroker(brokerURL).
		SetClientID("edgesense-agent-uplink").
		SetAutoReconnect(true).
		SetConnectRetry(true).
		SetConnectRetryInterval(2 * time.Second).
		SetMaxReconnectInterval(10 * time.Second).
		SetOrderMatters(false).
		SetOnConnectHandler(func(_ mqtt.Client) {
			log.Printf("uplink connected: %s", brokerURL)
			if onUp != nil {
				go onUp()
			}
		}).
		SetConnectionLostHandler(func(_ mqtt.Client, err error) {
			log.Printf("uplink connection lost: %v (events will be buffered)", err)
		})
	return &mqttUplink{client: mqtt.NewClient(opts), owned: true}
}

// sharedMQTTUplink wraps an externally managed client (single-broker layout,
// where sensors and events share the local broker connection).
func sharedMQTTUplink(c mqtt.Client) *mqttUplink {
	return &mqttUplink{client: c}
}

func (m *mqttUplink) Publish(ev event) error {
	payload, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	// Gate on a live connection: paho would otherwise queue the message
	// internally while reconnecting AND we'd buffer it, delivering it
	// twice after an outage. The disk buffer owns offline events.
	if !m.client.IsConnectionOpen() {
		return fmt.Errorf("uplink not connected")
	}
	topic := fmt.Sprintf("edgesense/events/%s", ev.MachineID)
	tok := m.client.Publish(topic, 1, false, payload)
	if !tok.WaitTimeout(publishTimeout) {
		return fmt.Errorf("publish timeout on %s", topic)
	}
	return tok.Error()
}

func (m *mqttUplink) Connected() bool { return m.client.IsConnectionOpen() }

func (m *mqttUplink) Start() {
	if m.owned {
		// ConnectRetry keeps trying in the background; failure here is
		// survivable — events buffer until the uplink comes up.
		m.client.Connect()
	}
}

func (m *mqttUplink) Close() {
	if m.owned {
		m.client.Disconnect(250)
	}
}
