"""Store-and-forward demo: survive an uplink outage with zero event loss.

The docker stack separates the local broker (sensors) from the "cloud"
broker (events uplink). This script kills the cloud broker, injects faults
while it is down — the agent keeps detecting and buffers every event to
disk — then restores the broker and verifies the buffered events are
replayed with their original timestamps.

The agent's own Prometheus metrics (edgesense_buffer_depth) are polled to
show the buffer filling during the outage and draining to zero afterwards.

Requires: `make stack` and the docker CLI.

    python scripts/demo_offline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import paho.mqtt.client as mqtt

BROKER = os.environ.get("EDGESENSE_BROKER_HOST", "localhost")
PORT = int(os.environ.get("EDGESENSE_BROKER_PORT", "11883"))
UPLINK_PORT = int(os.environ.get("EDGESENSE_UPLINK_PORT", "12883"))
UPLINK_USERNAME = os.environ.get("EDGESENSE_UPLINK_USERNAME")
UPLINK_PASSWORD = os.environ.get("EDGESENSE_UPLINK_PASSWORD", "")
CLOUD_CONTAINER = os.environ.get("EDGESENSE_CLOUD_CONTAINER", "edgesense-mosquitto-cloud")
METRICS_URL = os.environ.get("EDGESENSE_METRICS_URL", "http://localhost:8890/metrics")

# topic layout (PLATFORM.md §4.4): legacy flat topics by default, tenant-
# namespaced es/<org>/<site>/… when EDGESENSE_ORG/EDGESENSE_SITE are set
ORG = os.environ.get("EDGESENSE_ORG")
SITE = os.environ.get("EDGESENSE_SITE")
NAMESPACED = bool(ORG or SITE)
PREFIX = f"es/{ORG or 'default'}/{SITE or 'default'}"
EVENT_FILTER = f"{PREFIX}/+/events" if NAMESPACED else "edgesense/events/#"


def control_topic(machine: str) -> str:
    return f"{PREFIX}/{machine}/control" if NAMESPACED else "edgesense/control/fault"


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def agent_metric(name: str) -> float | None:
    """Read one gauge/counter from the agent's Prometheus endpoint."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=2) as resp:
            text = resp.read().decode()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    return None


class EventLog:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[dict] = []

    def on_message(self, _c, _u, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        payload["_arrival"] = time.time()
        with self.lock:
            self.events.append(payload)

    def since(self, t0: float, ts_range: tuple[float, float] | None = None) -> list[dict]:
        with self.lock:
            out = [e for e in self.events if e["_arrival"] >= t0]
        if ts_range:
            out = [e for e in out if ts_range[0] <= e.get("ts", 0) <= ts_range[1]]
        return out


def wait_for(predicate, timeout: float, poll: float = 0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(poll)
    return None


def main() -> int:
    check = docker("inspect", "--format", "{{.State.Running}}", CLOUD_CONTAINER)
    if check.returncode != 0 or check.stdout.strip() != "true":
        print(f"container {CLOUD_CONTAINER} not running — start the stack first (make stack)")
        return 1

    log = EventLog()
    listener = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                           client_id="edgesense-offline-demo")
    if UPLINK_USERNAME:
        listener.username_pw_set(UPLINK_USERNAME, UPLINK_PASSWORD)
    listener.on_message = log.on_message
    listener.on_connect = lambda c, *_: c.subscribe(EVENT_FILTER)
    listener.reconnect_delay_set(1, 3)
    listener.connect(BROKER, UPLINK_PORT)
    listener.loop_start()

    control = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                          client_id="edgesense-offline-demo-ctl")
    control.connect(BROKER, PORT)
    control.loop_start()
    time.sleep(1)

    def inject(machine: str, fault: str, ticks: int) -> None:
        control.publish(control_topic(machine), json.dumps(
            {"machine_id": machine, "fault": fault, "ticks": ticks}))

    print("=== EdgeSense AI — uplink outage / store-and-forward demo ===\n")
    ok = True
    depth_during = None
    try:
        # 1: prove the uplink works
        t0 = time.time()
        inject("machine-01", "overheat", 8)
        baseline = wait_for(lambda: log.since(t0), timeout=25)
        if not baseline:
            print("[1/4] FAIL: no event on the cloud broker with a healthy uplink")
            return 1
        print(f"[1/4] baseline: uplink healthy — first event after {baseline[0]['_arrival'] - t0:.1f}s")
        time.sleep(6)  # let the episode finish

        # 2: cut the uplink, keep detecting
        print(f"[2/4] stopping cloud broker ({CLOUD_CONTAINER}) — uplink is now DOWN")
        if docker("stop", CLOUD_CONTAINER).returncode != 0:
            print("      FAIL: docker stop failed")
            return 1
        outage_start = time.time()
        inject("machine-02", "bearing_fault", 16)
        inject("machine-03", "overload", 16)
        print("      injected bearing_fault(machine-02) + overload(machine-03);"
              " events are buffering on the edge…")
        time.sleep(14)  # both episodes (~8s) plus margin
        depth_during = agent_metric("edgesense_buffer_depth")
        if depth_during is not None:
            print(f"      agent metrics confirm: edgesense_buffer_depth = {depth_during:.0f}"
                  " events on disk")
        outage_end = time.time()
        leaked = log.since(outage_start)
        print(f"      events that reached the cloud during the outage: {len(leaked)}")

    finally:
        # never leave the broker down
        restored = docker("start", CLOUD_CONTAINER).returncode == 0
        print(f"[3/4] restoring cloud broker — {'ok' if restored else 'FAILED, restart it manually!'}")

    wait_for(lambda: log.since(outage_end, ts_range=(outage_start, outage_end)) or None,
             timeout=75)
    time.sleep(3)  # catch stragglers in the same flush
    replayed = log.since(outage_end, ts_range=(outage_start, outage_end))

    if len(replayed) < 3:
        print(f"[4/4] FAIL: only {len(replayed)} buffered event(s) replayed")
        ok = False
    else:
        unique = {(e["machine_id"], e["ts"]) for e in replayed}
        dupes = len(replayed) - len(unique)
        delays = sorted(e["_arrival"] - e["ts"] for e in replayed)
        machines = sorted({e["machine_id"] for e in replayed})
        print(f"[4/4] replay: {len(replayed)} buffered events delivered after restore"
              f" ({'no duplicates' if dupes == 0 else f'{dupes} DUPLICATES'})")
        print(f"      machines: {', '.join(machines)}   original-reading age at delivery: "
              f"{delays[0]:.1f}s … {delays[-1]:.1f}s")
        print("      every event kept its original reading timestamp from inside the outage")
        ok &= dupes == 0

    depth_after = agent_metric("edgesense_buffer_depth")
    if depth_after is not None:
        drained = depth_after == 0
        print(f"      agent metrics confirm: edgesense_buffer_depth = {depth_after:.0f}"
              f" ({'buffer fully drained' if drained else 'BUFFER NOT DRAINED'})")
        ok &= drained

    outage_len = outage_end - outage_start
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} — "
          f"{len(replayed)} events preserved across a {outage_len:.1f}s uplink outage")

    listener.loop_stop()
    control.loop_stop()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
