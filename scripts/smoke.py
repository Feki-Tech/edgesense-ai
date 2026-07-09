"""End-to-end smoke test.

Requires: broker (make broker), inference (make inference), agent (make agent).
Publishes one healthy and one clearly faulty reading, then asserts that the
agent emits an anomaly event for the faulty one only.
"""

from __future__ import annotations

import json
import sys
import time

import paho.mqtt.client as mqtt
import requests

BROKER, PORT = "localhost", 11883
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
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edgesense-smoke")

    def on_message(_c, _u, msg) -> None:
        events.append(json.loads(msg.payload))

    client.on_message = on_message
    client.connect(BROKER, PORT)
    client.subscribe(f"edgesense/events/{MACHINE}")
    client.loop_start()
    time.sleep(0.5)

    client.publish(f"edgesense/sensors/{MACHINE}", json.dumps(HEALTHY))
    client.publish(f"edgesense/sensors/{MACHINE}", json.dumps(FAULTY))

    deadline = time.time() + 10
    while time.time() < deadline and not events:
        time.sleep(0.2)
    client.loop_stop()

    got_anomaly = len(events) == 1 and events[0]["reading"]["vibration"] == FAULTY["vibration"]
    print(f"[3/3] agent round-trip: {len(events)} event(s) received "
          f"-> {'OK' if got_anomaly else 'FAIL'}")
    ok &= got_anomaly

    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
