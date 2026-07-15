// The /events CoAP resource: decode, validate, republish to MQTT.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/fxamacker/cbor/v2"
	"github.com/plgd-dev/go-coap/v3/message"
	"github.com/plgd-dev/go-coap/v3/message/codes"
	coapmux "github.com/plgd-dev/go-coap/v3/mux"
)

const eventsPath = "/events"

// event mirrors the edge agent's anomaly event (JSON wire names are shared
// by both the CBOR and JSON encodings).
type event struct {
	MachineID string  `json:"machine_id"`
	Ts        float64 `json:"ts"`
	Score     float64 `json:"score"`
	Reason    string  `json:"reason,omitempty"`
	Reading   reading `json:"reading"`
	AgentTs   float64 `json:"agent_ts"`
}

type reading struct {
	MachineID   string  `json:"machine_id"`
	Ts          float64 `json:"ts"`
	Vibration   float64 `json:"vibration"`
	Temperature float64 `json:"temperature"`
	Current     float64 `json:"current"`
}

// decodeEvent parses a payload by declared content format: CBOR (what the
// agent sends) or JSON (fallback for standard CoAP tooling). Without a
// content-format option both are tried.
func decodeEvent(body []byte, cf message.MediaType, haveCF bool) (event, error) {
	var ev event
	switch {
	case haveCF && cf == message.AppCBOR:
		return ev, cbor.Unmarshal(body, &ev)
	case haveCF && cf == message.AppJSON:
		return ev, json.Unmarshal(body, &ev)
	case haveCF:
		return ev, fmt.Errorf("unsupported content format %v", cf)
	default:
		if jerr := json.Unmarshal(body, &ev); jerr == nil {
			return ev, nil
		}
		return ev, cbor.Unmarshal(body, &ev)
	}
}

// validate rejects events that cannot form a safe MQTT topic.
func validate(ev event) error {
	if ev.MachineID == "" {
		return fmt.Errorf("missing machine_id")
	}
	if strings.ContainsAny(ev.MachineID, "/+#\x00") {
		return fmt.Errorf("invalid machine_id %q", ev.MachineID)
	}
	return nil
}

// eventsHandler bridges POSTed events to the publish function (MQTT QoS 1).
// It replies 2.04 only after the publish is acknowledged, 4.xx for payloads
// that can never succeed, and 5.03 when the broker is unavailable so the
// agent keeps the event buffered and retries.
func eventsHandler(publish func(topic string, payload []byte) error) coapmux.Handler {
	return coapmux.HandlerFunc(func(w coapmux.ResponseWriter, r *coapmux.Message) {
		if r.Code() != codes.POST {
			_ = w.SetResponse(codes.MethodNotAllowed, message.TextPlain, nil)
			return
		}
		eventsReceived.Inc()

		body, err := r.ReadBody()
		if err != nil || len(body) == 0 {
			eventsRejected.Inc()
			_ = w.SetResponse(codes.BadRequest, message.TextPlain, nil)
			return
		}
		cf, cfErr := r.ContentFormat()
		ev, err := decodeEvent(body, cf, cfErr == nil)
		if err == nil {
			err = validate(ev)
		}
		if err != nil {
			eventsRejected.Inc()
			log.Printf("rejected event from %v: %v", w.Conn().RemoteAddr(), err)
			_ = w.SetResponse(codes.BadRequest, message.TextPlain, nil)
			return
		}

		payload, err := json.Marshal(ev) // canonical JSON for MQTT consumers
		if err != nil {
			eventsRejected.Inc()
			_ = w.SetResponse(codes.InternalServerError, message.TextPlain, nil)
			return
		}
		topic := "edgesense/events/" + ev.MachineID
		if err := publish(topic, payload); err != nil {
			republishFailures.Inc()
			log.Printf("republish %s failed: %v (agent will buffer)", topic, err)
			_ = w.SetResponse(codes.ServiceUnavailable, message.TextPlain, nil)
			return
		}
		eventsRepublished.Inc()
		_ = w.SetResponse(codes.Changed, message.TextPlain, nil)
	})
}
