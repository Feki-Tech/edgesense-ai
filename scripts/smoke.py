"""End-to-end smoke test.

Requires the stack (make stack, or broker+inference+agent locally).
Publishes one healthy and one clearly faulty reading, then asserts that the
agent emits an anomaly event for the faulty one only. Events are awaited on
the uplink broker (12883) and the local broker (11883) — whichever is around.
"""

from __future__ import annotations

import json
import os
import sys
import time

import paho.mqtt.client as mqtt
import requests

BROKER = os.environ.get("EDGESENSE_BROKER_HOST", "localhost")
PORT = int(os.environ.get("EDGESENSE_BROKER_PORT", "11883"))
UPLINK_PORT = int(os.environ.get("EDGESENSE_UPLINK_PORT", "12883"))
INFERENCE = "http://localhost:8800"
MACHINE = "machine-smoke"

HEALTHY = {"machine_id": MACHINE, "ts": time.time(),
           "vibration": 0.8, "temperature": 45.0, "current": 12.0}
FAULTY = {"machine_id": MACHINE, "ts": time.time(),
          "vibration": 4.2, "temperature": 46.0, "current": 14.5}


def main() -> int:
    ok = True

    health = requests.get(f"{INFERENCE}/healthz", timeout=3).json()
    print(f"[1/3] inference healthy: {health['status']}")

    direct = requests.post(f"{INFERENCE}/score", json={
        k: FAULTY[k] for k in ("vibration", "temperature", "current")}, timeout=3).json()
    print(f"[2/3] direct score of faulty reading: {direct}")
    ok &= direct["is_anomaly"]

    events: list[dict] = []

    def on_message(_c, _u, msg) -> None:
        events.append(json.loads(msg.payload))

    listeners: list[mqtt.Client] = []
    for i, port in enumerate(dict.fromkeys((UPLINK_PORT, PORT))):
        listener = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                               client_id=f"edgesense-smoke-sub-{i}")
        listener.on_message = on_message
        try:
            listener.connect(BROKER, port)
        except OSError:
            continue  # that broker isn't running in this setup
        listener.subscribe(f"edgesense/events/{MACHINE}")
        listener.loop_start()
        listeners.append(listener)

    if not listeners:
        print("no event broker reachable")
        return 1

    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edgesense-smoke-pub")
    pub.connect(BROKER, PORT)
    pub.loop_start()
    time.sleep(0.5)

    pub.publish(f"edgesense/sensors/{MACHINE}", json.dumps(HEALTHY))
    pub.publish(f"edgesense/sensors/{MACHINE}", json.dumps(FAULTY))

    deadline = time.time() + 10
    while time.time() < deadline and not events:
        time.sleep(0.2)
    for listener in listeners:
        listener.loop_stop()
    pub.loop_stop()

    got_anomaly = (len(events) >= 1
                   and all(ev["reading"]["vibration"] == FAULTY["vibration"] for ev in events))
    print(f"[3/3] agent round-trip: {len(events)} event(s) received "
          f"-> {'OK' if got_anomaly else 'FAIL'}")
    ok &= got_anomaly

    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
