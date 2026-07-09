"""Simulator behavior tests (no MQTT needed)."""

from __future__ import annotations

import random

from simulator.simulate import FAULT_TYPES, Machine


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
