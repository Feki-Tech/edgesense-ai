package main

import (
	"errors"
	"path/filepath"
	"testing"
)

func newTestBuffer(t *testing.T, max int) *EventBuffer {
	t.Helper()
	return newEventBuffer(filepath.Join(t.TempDir(), "buf.jsonl"), max)
}

func ev(id string, ts float64) event {
	return event{MachineID: id, Ts: ts, Score: -0.1}
}

func TestAddLenAndPersistence(t *testing.T) {
	b := newTestBuffer(t, 100)
	for i, id := range []string{"m1", "m2", "m3"} {
		if err := b.Add(ev(id, float64(i))); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if got := b.Len(); got != 3 {
		t.Fatalf("Len = %d, want 3", got)
	}

	// a fresh instance on the same path must see the same events
	b2 := newEventBuffer(b.path, 100)
	if got := b2.Len(); got != 3 {
		t.Fatalf("persisted Len = %d, want 3", got)
	}
}

func TestDrainFIFOAndEmpties(t *testing.T) {
	b := newTestBuffer(t, 100)
	for i, id := range []string{"a", "b", "c"} {
		if err := b.Add(ev(id, float64(i))); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	var order []string
	n, err := b.DrainTo(func(e event) error {
		order = append(order, e.MachineID)
		return nil
	})
	if err != nil || n != 3 {
		t.Fatalf("DrainTo = (%d, %v), want (3, nil)", n, err)
	}
	if len(order) != 3 || order[0] != "a" || order[1] != "b" || order[2] != "c" {
		t.Fatalf("wrong order: %v", order)
	}
	if got := b.Len(); got != 0 {
		t.Fatalf("Len after drain = %d, want 0", got)
	}
}

func TestDrainPartialFailureKeepsRemainder(t *testing.T) {
	b := newTestBuffer(t, 100)
	for i, id := range []string{"a", "b", "c"} {
		if err := b.Add(ev(id, float64(i))); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	calls := 0
	n, err := b.DrainTo(func(e event) error {
		calls++
		if calls == 2 {
			return errors.New("broker gone")
		}
		return nil
	})
	if n != 1 || err == nil {
		t.Fatalf("DrainTo = (%d, %v), want (1, error)", n, err)
	}
	if got := b.Len(); got != 2 {
		t.Fatalf("Len after partial drain = %d, want 2", got)
	}

	var order []string
	n, err = b.DrainTo(func(e event) error {
		order = append(order, e.MachineID)
		return nil
	})
	if err != nil || n != 2 {
		t.Fatalf("second DrainTo = (%d, %v), want (2, nil)", n, err)
	}
	if order[0] != "b" || order[1] != "c" {
		t.Fatalf("remainder order wrong: %v", order)
	}
}

func TestCapDropsOldest(t *testing.T) {
	b := newTestBuffer(t, 3)
	for i, id := range []string{"a", "b", "c", "d"} {
		if err := b.Add(ev(id, float64(i))); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if got := b.Len(); got != 3 {
		t.Fatalf("Len = %d, want 3", got)
	}
	var order []string
	if _, err := b.DrainTo(func(e event) error {
		order = append(order, e.MachineID)
		return nil
	}); err != nil {
		t.Fatalf("DrainTo: %v", err)
	}
	if order[0] != "b" || order[2] != "d" {
		t.Fatalf("expected oldest dropped, got %v", order)
	}
}
