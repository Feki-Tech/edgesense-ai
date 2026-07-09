package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// EventBuffer is a disk-backed FIFO (JSON Lines) for anomaly events that
// could not be published, e.g. while the broker is unreachable. Events
// survive agent restarts. When the buffer is full the oldest events are
// dropped first.
//
// Publishing while draining holds the lock, so Add blocks during a drain;
// acceptable at edge event rates (events are rare by design).
type EventBuffer struct {
	mu   sync.Mutex
	path string
	max  int
}

func newEventBuffer(path string, max int) *EventBuffer {
	return &EventBuffer{path: path, max: max}
}

// Add appends an event, dropping the oldest entries beyond capacity.
func (b *EventBuffer) Add(ev event) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	evs, err := b.read()
	if err != nil {
		return err
	}
	evs = append(evs, ev)
	if len(evs) > b.max {
		evs = evs[len(evs)-b.max:]
	}
	return b.write(evs)
}

// Len reports the number of buffered events.
func (b *EventBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	evs, err := b.read()
	if err != nil {
		return 0
	}
	return len(evs)
}

// DrainTo publishes buffered events in FIFO order. It stops at the first
// failure, keeps unpublished events (including the failed one) buffered,
// and returns the number of events successfully published.
func (b *EventBuffer) DrainTo(publish func(event) error) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	evs, err := b.read()
	if err != nil || len(evs) == 0 {
		return 0, err
	}
	for i, ev := range evs {
		if perr := publish(ev); perr != nil {
			if werr := b.write(evs[i:]); werr != nil {
				return i, fmt.Errorf("publish: %v; buffer rewrite: %w", perr, werr)
			}
			return i, perr
		}
	}
	return len(evs), b.write(nil)
}

func (b *EventBuffer) read() ([]event, error) {
	f, err := os.Open(b.path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var evs []event
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var ev event
		if json.Unmarshal(line, &ev) == nil {
			evs = append(evs, ev)
		}
	}
	return evs, sc.Err()
}

// write atomically replaces the buffer file; an empty slice removes it.
func (b *EventBuffer) write(evs []event) error {
	if len(evs) == 0 {
		err := os.Remove(b.path)
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if dir := filepath.Dir(b.path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := b.path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	for _, ev := range evs {
		if err := enc.Encode(ev); err != nil {
			f.Close()
			os.Remove(tmp)
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, b.path)
}
