"""Simulator behavior tests (no MQTT needed)."""

from __future__ import annotations

import json
import random

from simulator.simulate import FAULT_TYPES, Machine, apply_control


def make_machine(seed: int = 42) -> Machine:
    return Machine(machine_id="m-test", rng=random.Random(seed))


def test_healthy_machine_stays_in_normal_band() -> None:
    m = make_machine()
    for _ in range(300):
        r = m.step(anomaly_prob=0.0)
        assert r["fault_injected"] is None
        assert 0.0 <= r["vibration"] < 2.0
        assert 38.0 < r["temperature"] < 52.0
        assert 8.0 < r["current"] < 16.0


def test_fault_episode_starts_and_clears() -> None:
    m = make_machine()
    r = m.step(anomaly_prob=1.0)
    assert m.fault in FAULT_TYPES or r["fault_injected"] in FAULT_TYPES

    ticks = 0
    while m.fault is not None and ticks < 100:
        m.step(anomaly_prob=0.0)
        ticks += 1
    assert m.fault is None, "fault episode never cleared"
    assert ticks <= 40


def test_bearing_fault_raises_vibration() -> None:
    m = make_machine()
    m.fault = "bearing_fault"
    m.fault_ticks_left = 10
    r = m.step(anomaly_prob=0.0)
    assert r["vibration"] > 1.5


def test_reading_schema() -> None:
    r = make_machine().step(anomaly_prob=0.0)
    assert set(r) == {"machine_id", "ts", "vibration", "temperature", "current", "fault_injected"}


def test_last_faulty_reading_keeps_its_label() -> None:
    # the reading generated on the final tick is shaped by the fault and must
    # be labeled with it, even though the episode clears in the same step
    m = make_machine()
    m.start_fault("overload", 1)
    r = m.step(anomaly_prob=0.0)
    assert r["fault_injected"] == "overload"
    assert m.fault is None
    assert m.step(anomaly_prob=0.0)["fault_injected"] is None


def test_control_injects_fault() -> None:
    m = make_machine()
    msg = apply_control({"m-test": m}, json.dumps(
        {"machine_id": "m-test", "fault": "overheat", "ticks": 5}))
    assert m.fault == "overheat"
    assert m.fault_ticks_left == 5
    assert "overheat" in msg


def test_control_clear_ends_episode() -> None:
    m = make_machine()
    m.start_fault("overload", 30)
    apply_control({"m-test": m}, json.dumps({"machine_id": "m-test", "fault": "clear"}))
    assert m.fault is None
    assert m.step(anomaly_prob=0.0)["fault_injected"] is None


def test_control_ignores_garbage() -> None:
    m = make_machine()
    fleet = {"m-test": m}
    for payload in (b"not json", b"[1,2]",
                    json.dumps({"machine_id": "nope", "fault": "overheat"}),
                    json.dumps({"machine_id": "m-test", "fault": "explode"})):
        out = apply_control(fleet, payload)
        assert "ignored" in out
    assert m.fault is None
