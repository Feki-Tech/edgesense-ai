package main

import "testing"

func TestMachineIDFromTopic(t *testing.T) {
	cases := map[string]string{
		// legacy layout: last segment
		"edgesense/sensors/machine-01": "machine-01",
		"edgesense/sensors/a/b":        "b",
		"bare":                         "bare",
		// namespaced layout: 4th segment
		"es/acme-pumps/lyon-plant/pump-07/sensors/vibration": "pump-07",
		"es/default/default/machine-01/sensors":              "machine-01",
		"es/default/default/machine-02/events":               "machine-02",
		"es/org/site/m1/control":                             "m1",
		"es/org/site/m1":                                     "m1",
		// too short to be namespaced: fall back to last segment
		"es/org/site": "site",
	}
	for topic, want := range cases {
		if got := machineIDFromTopic(topic); got != want {
			t.Errorf("machineIDFromTopic(%q) = %q, want %q", topic, got, want)
		}
	}
}

func TestLayoutFromEnvLegacyDefault(t *testing.T) {
	t.Setenv("EDGESENSE_ORG", "")
	t.Setenv("EDGESENSE_SITE", "")
	l := layoutFromEnv()
	if l.namespaced {
		t.Fatalf("expected legacy layout with no env set, got %+v", l)
	}
	if got := l.eventTopic("machine-01"); got != "edgesense/events/machine-01" {
		t.Errorf("eventTopic = %q", got)
	}
	if got := l.defaultSensorFilter(); got != "edgesense/sensors/#" {
		t.Errorf("defaultSensorFilter = %q", got)
	}
}

func TestLayoutFromEnvNamespaced(t *testing.T) {
	t.Setenv("EDGESENSE_ORG", "acme-pumps")
	t.Setenv("EDGESENSE_SITE", "lyon-plant")
	l := layoutFromEnv()
	if !l.namespaced || l.org != "acme-pumps" || l.site != "lyon-plant" {
		t.Fatalf("unexpected layout: %+v", l)
	}
	if got := l.eventTopic("pump-07"); got != "es/acme-pumps/lyon-plant/pump-07/events" {
		t.Errorf("eventTopic = %q", got)
	}
	if got := l.defaultSensorFilter(); got != "es/acme-pumps/lyon-plant/+/sensors/#" {
		t.Errorf("defaultSensorFilter = %q", got)
	}
}

func TestLayoutFromEnvPartialDefaults(t *testing.T) {
	t.Setenv("EDGESENSE_ORG", "acme-pumps")
	t.Setenv("EDGESENSE_SITE", "")
	l := layoutFromEnv()
	if !l.namespaced || l.org != "acme-pumps" || l.site != "default" {
		t.Fatalf("unexpected layout: %+v", l)
	}

	t.Setenv("EDGESENSE_ORG", "")
	t.Setenv("EDGESENSE_SITE", "lyon-plant")
	l = layoutFromEnv()
	if !l.namespaced || l.org != "default" || l.site != "lyon-plant" {
		t.Fatalf("unexpected layout: %+v", l)
	}
}
