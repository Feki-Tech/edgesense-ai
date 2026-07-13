"""Live fault-injection demo against a running EdgeSense stack.

Walks three predictive-maintenance scenarios end to end: injects a bearing
fault, an overheat and an overload into the running simulator via the MQTT
control topic, then measures — against ground truth from the sensor stream —
how fast the edge agent raises each alarm and whether any healthy reading
was flagged.

Requires: `make stack` (or local broker+inference+agent+simulator).

    python scripts/demo.py [--ticks 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict

import paho.mqtt.client as mqtt

BROKER = os.environ.get("EDGESENSE_BROKER_HOST", "localhost")
PORT = int(os.environ.get("EDGESENSE_BROKER_PORT", "11883"))
UPLINK_PORT = int(os.environ.get("EDGESENSE_UPLINK_PORT", "12883"))
CONTROL_TOPIC = "edgesense/control/fault"

SCENARIOS = [
    ("bearing_fault", "worn bearing → vibration 3–5×, current +10–25%"),
    ("overheat", "cooling failure → temperature ramps +15→+30°C"),
    ("overload", "mechanical overload → current 1.6–2×, vibration 1.4–1.8×"),
]


class Feed:
    """Collects ground truth (sensor stream) and anomaly events."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.readings: dict[str, list[dict]] = defaultdict(list)
        self.events: list[dict] = []
        self._seen: set[tuple[str, float]] = set()

    def on_message(self, _c, _u, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        with self.lock:
            if msg.topic.startswith("edgesense/sensors/"):
                self.readings[payload.get("machine_id", "?")].append(payload)
            elif msg.topic.startswith("edgesense/events/"):
                key = (payload.get("machine_id", "?"), payload.get("ts", 0.0))
                if key not in self._seen:  # events may arrive on both brokers
                    self._seen.add(key)
                    payload["_arrival"] = time.time()
                    self.events.append(payload)

    def machine_readings(self, machine: str) -> list[dict]:
        with self.lock:
            return list(self.readings[machine])

    def machine_events(self, machine: str, t0: float, t1: float) -> list[dict]:
        with self.lock:
            return [e for e in self.events
                    if e.get("machine_id") == machine and t0 <= e.get("ts", 0) <= t1]

    def all_events(self) -> list[dict]:
        with self.lock:
            return list(self.events)


def wait_for(predicate, timeout: float, poll: float = 0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(poll)
    return None


def connect(host: str, port: int, feed: Feed, topics: list[str],
            client_id: str) -> mqtt.Client | None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.on_message = feed.on_message
    client.on_connect = lambda c, *_: c.subscribe([(t, 0) for t in topics])
    try:
        client.connect(host, port)
    except OSError:
        return None
    client.loop_start()
    return client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=24, help="fault episode length")
    args = ap.parse_args()

    feed = Feed()
    clients = []
    local = connect(BROKER, PORT, feed,
                    ["edgesense/sensors/#", "edgesense/events/#"], "edgesense-demo-local")
    if local is None:
        print(f"cannot reach broker at {BROKER}:{PORT} — is the stack up? (make stack)")
        return 1
    clients.append(local)
    if UPLINK_PORT != PORT:
        uplink = connect(BROKER, UPLINK_PORT, feed,
                         ["edgesense/events/#"], "edgesense-demo-uplink")
        if uplink:
            clients.append(uplink)

    print("=== EdgeSense AI — live fault-injection demo ===")
    print(f"sensors/control: {BROKER}:{PORT}   events: "
          f"{', '.join(str(p) for p in dict.fromkeys((PORT, UPLINK_PORT)))}\n")

    machines = wait_for(lambda: sorted(feed.readings)[:3] if len(feed.readings) >= 3 else None,
                        timeout=15)
    if not machines:
        print("no sensor data seen within 15s — is the simulator running?")
        return 1
    print(f"machines online: {', '.join(machines)}")

    rows = []
    ok = True
    for i, ((fault, blurb), machine) in enumerate(zip(SCENARIOS, machines), start=1):
        print(f"\n--- scenario {i}/{len(SCENARIOS)}: {fault} on {machine} ---")
        print(f"    {blurb}")
        base = len(feed.machine_readings(machine))
        local.publish(CONTROL_TOPIC, json.dumps(
            {"machine_id": machine, "fault": fault, "ticks": args.ticks}))

        def fault_bounds():
            readings = feed.machine_readings(machine)[base:]
            start = next((r for r in readings if r.get("fault_injected") == fault), None)
            if start is None:
                return None
            end = next((r for r in readings
                        if r["ts"] > start["ts"] and r.get("fault_injected") is None), None)
            return (start, end) if end is not None else None

        bounds = wait_for(fault_bounds, timeout=args.ticks * 0.5 + 30)
        if bounds is None:
            print("    FAIL: fault episode never observed in sensor stream")
            ok = False
            continue
        start, end = bounds
        time.sleep(2)  # let in-flight events land

        events = feed.machine_events(machine, start["ts"] - 0.25, end["ts"] - 0.01)
        if not events:
            print("    FAIL: no anomaly event raised")
            rows.append((fault, machine, "MISSED", "—", "—", "—"))
            ok = False
            continue

        first = min(events, key=lambda e: e["ts"])
        latency_s = first["ts"] - start["ts"]
        faulty_seen = [r for r in feed.machine_readings(machine)
                       if start["ts"] <= r["ts"] < end["ts"]]
        nth = sum(1 for r in faulty_seen if r["ts"] <= first["ts"])
        reasons = defaultdict(int)
        for e in events:
            reasons[e.get("reason", "?")] += 1
        reason_str = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items()))

        print(f"    detected on reading #{nth} after {latency_s:.2f}s — "
              f"{len(events)}/{len(faulty_seen)} faulty readings flagged")
        print(f"    trigger reasons: {reason_str}")
        rows.append((fault, machine, f"{latency_s:.2f}s (reading #{nth})",
                     f"{len(events)}/{len(faulty_seen)}", reason_str, "PASS"))

    # false-positive audit: every event must map to a ground-truth faulty reading
    fp = 0
    unmatched = 0
    healthy_total = 0
    with feed.lock:
        all_readings = [r for rs in feed.readings.values() for r in rs]
    healthy_total = sum(1 for r in all_readings if r.get("fault_injected") is None)
    by_machine: dict[str, list[dict]] = defaultdict(list)
    for r in all_readings:
        by_machine[r["machine_id"]].append(r)
    for e in feed.all_events():
        candidates = by_machine.get(e.get("machine_id", "?"), [])
        match = next((r for r in candidates if abs(r["ts"] - e.get("ts", 0)) < 0.05), None)
        if match is None:
            unmatched += 1
        elif match.get("fault_injected") is None:
            fp += 1

    print("\n=== summary ===")
    widths = (14, 12, 24, 10, 26, 6)
    header = ("fault", "machine", "time-to-detect", "flagged", "reasons", "result")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    fp_rate = fp / healthy_total if healthy_total else 0.0
    print(f"\nfalse positives: {fp} across {healthy_total} healthy readings "
          f"({fp_rate:.1%}; model is tuned for ~0.5%)"
          + (f" ({unmatched} events outside capture window ignored)" if unmatched else ""))
    ok &= fp_rate <= 0.02  # gross-misbehavior gate, not a zero-FP demand

    verdict = "PASS" if ok else "FAIL"
    print(f"VERDICT: {verdict}")

    for c in clients:
        c.loop_stop()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
