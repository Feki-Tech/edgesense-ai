"""Machine sensor simulator.

Publishes vibration / temperature / current readings for N virtual machines
to MQTT (topic: edgesense/sensors/<machine_id>). Occasionally injects fault
episodes so the anomaly detector has something to find.

Faults can also be injected on demand (for demos and testing) by publishing
a JSON command to `edgesense/control/fault`:

    {"machine_id": "machine-01", "fault": "bearing_fault", "ticks": 24}

`fault` is one of bearing_fault / overheat / overload, or "clear" to end the
current episode immediately.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import signal
import sys
import time
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt

FAULT_TYPES = ("bearing_fault", "overheat", "overload")
CONTROL_TOPIC = "edgesense/control/fault"


@dataclass
class Machine:
    machine_id: str
    temperature: float = 45.0   # °C
    vibration: float = 0.8      # mm/s RMS
    current: float = 12.0       # A
    fault: str | None = None
    fault_ticks_left: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def start_fault(self, fault: str, ticks: int) -> None:
        self.fault = fault
        self.fault_ticks_left = max(1, ticks)
        print(f"[{self.machine_id}] !! injected fault: {fault} "
              f"({self.fault_ticks_left} ticks)", flush=True)

    def maybe_start_fault(self, anomaly_prob: float) -> None:
        if self.fault is None and self.rng.random() < anomaly_prob:
            self.start_fault(self.rng.choice(FAULT_TYPES), self.rng.randint(20, 40))

    def step(self, anomaly_prob: float) -> dict:
        self.maybe_start_fault(anomaly_prob)

        # normal operating point with noise and slow drift
        temp = 45.0 + self.rng.gauss(0, 1.2)
        vib = 0.8 + self.rng.gauss(0, 0.15)
        cur = 12.0 + self.rng.gauss(0, 0.6)

        if self.fault == "bearing_fault":
            vib *= self.rng.uniform(3.0, 5.0)
            cur *= self.rng.uniform(1.1, 1.25)
        elif self.fault == "overheat":
            progress = 1.0 - self.fault_ticks_left / 40.0
            temp += 15.0 + 15.0 * progress
        elif self.fault == "overload":
            cur *= self.rng.uniform(1.6, 2.0)
            vib *= self.rng.uniform(1.4, 1.8)

        active_fault = self.fault  # the fault that shaped THIS reading
        if self.fault is not None:
            self.fault_ticks_left -= 1
            if self.fault_ticks_left <= 0:
                print(f"[{self.machine_id}] fault cleared: {self.fault}", flush=True)
                self.fault = None

        self.temperature, self.vibration, self.current = temp, max(vib, 0.0), max(cur, 0.0)
        return {
            "machine_id": self.machine_id,
            "ts": time.time(),
            "vibration": round(self.vibration, 4),
            "temperature": round(self.temperature, 2),
            "current": round(self.current, 3),
            "fault_injected": active_fault,  # ground truth, for demo/debugging only
        }


def apply_control(machines: dict[str, Machine], payload: bytes | str) -> str:
    """Apply one control command to the fleet. Returns a log line."""
    try:
        cmd = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "control: ignored (bad JSON)"
    if not isinstance(cmd, dict):
        return "control: ignored (bad JSON)"
    machine = machines.get(cmd.get("machine_id", ""))
    if machine is None:
        return f"control: ignored (unknown machine {cmd.get('machine_id')!r})"
    fault = cmd.get("fault")
    if fault == "clear":
        machine.fault, machine.fault_ticks_left = None, 0
        return f"control: {machine.machine_id} cleared"
    if fault not in FAULT_TYPES:
        return f"control: ignored (unknown fault {fault!r})"
    machine.start_fault(fault, int(cmd.get("ticks", 30)))
    return f"control: {machine.machine_id} -> {fault}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=11883)
    ap.add_argument("--machines", type=int, default=3)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between readings")
    ap.add_argument("--anomaly-prob", type=float, default=0.01,
                    help="per-tick probability a machine starts a fault episode")
    args = ap.parse_args()

    machines = {f"machine-{i:02d}": Machine(machine_id=f"machine-{i:02d}")
                for i in range(1, args.machines + 1)}
    control_q: queue.Queue[bytes] = queue.Queue()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edgesense-simulator")
    client.on_connect = lambda c, *_: c.subscribe(CONTROL_TOPIC)
    client.on_message = lambda _c, _u, msg: control_q.put(msg.payload)
    client.connect(args.broker, args.port)
    client.loop_start()

    print(f"simulating {len(machines)} machines -> mqtt://{args.broker}:{args.port} "
          f"(control: {CONTROL_TOPIC})", flush=True)

    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        while not control_q.empty():
            print(apply_control(machines, control_q.get_nowait()), flush=True)
        for m in machines.values():
            reading = m.step(args.anomaly_prob)
            client.publish(f"edgesense/sensors/{m.machine_id}", json.dumps(reading))
        time.sleep(args.interval)

    client.loop_stop()
    client.disconnect()
    print("simulator stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
