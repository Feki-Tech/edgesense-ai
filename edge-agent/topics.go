// Topic layout selection: legacy flat topics vs tenant-namespaced topics
// (PLATFORM.md §4.4, phase 1).
package main

import (
	"fmt"
	"os"
	"strings"
)

// topicLayout selects between the legacy flat layout
// (edgesense/{sensors,events}/<machine>) and the tenant-namespaced layout
// (es/<org>/<site>/<machine>/{sensors/…,events,control}).
//
// Namespacing is opt-in: enabled iff EDGESENSE_ORG or EDGESENSE_SITE is set;
// whichever is unset defaults to "default". With neither set the agent keeps
// today's single-tenant demo behavior unchanged.
type topicLayout struct {
	namespaced bool
	org, site  string
}

func layoutFromEnv() topicLayout {
	org, site := os.Getenv("EDGESENSE_ORG"), os.Getenv("EDGESENSE_SITE")
	if org == "" && site == "" {
		return topicLayout{}
	}
	if org == "" {
		org = "default"
	}
	if site == "" {
		site = "default"
	}
	return topicLayout{namespaced: true, org: org, site: site}
}

// eventTopic is the uplink topic an anomaly event for machine is published to.
func (l topicLayout) eventTopic(machine string) string {
	if l.namespaced {
		return fmt.Sprintf("es/%s/%s/%s/events", l.org, l.site, machine)
	}
	return fmt.Sprintf("edgesense/events/%s", machine)
}

// defaultSensorFilter is the subscription used when EDGESENSE_SENSOR_TOPIC is
// unset.
func (l topicLayout) defaultSensorFilter() string {
	if l.namespaced {
		return fmt.Sprintf("es/%s/%s/+/sensors/#", l.org, l.site)
	}
	return "edgesense/sensors/#"
}

// machineIDFromTopic extracts the machine id from a sensor topic in either
// layout: "edgesense/sensors/<machine>" (legacy: last segment) or
// "es/<org>/<site>/<machine>/sensors/…" (namespaced: 4th segment). Payloads
// carry machine_id, so this is only a fallback — but it must not mis-parse
// either layout.
func machineIDFromTopic(topic string) string {
	parts := strings.Split(topic, "/")
	if parts[0] == "es" && len(parts) >= 4 {
		return parts[3]
	}
	return parts[len(parts)-1]
}
