"""Simulator behavior tests (no MQTT needed)."""

from __future__ import annotations

import json
import random

from simulator.simulate import (FAULT_TYPES, Machine, apply_control,
                                control_filter, control_machine_id, sensor_topic)


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


def test_control_machine_from_topic_wins() -> None:
    # per-machine control: the topic addresses the machine (that is what the
    # broker ACLs scope), payload machine_id is ignored
    m = make_machine()
    apply_control({"m-test": m},
                  json.dumps({"machine_id": "someone-else", "fault": "overheat", "ticks": 4}),
                  machine_id="m-test")
    assert m.fault == "overheat"
    assert m.fault_ticks_left == 4


def test_control_topic_machine_unknown_is_ignored() -> None:
    m = make_machine()
    out = apply_control({"m-test": m},
                        json.dumps({"fault": "overheat"}), machine_id="ghost")
    assert "ignored" in out
    assert m.fault is None


def test_control_payload_machine_still_works_without_topic() -> None:
    # legacy global topic path: machine comes from the payload
    m = make_machine()
    apply_control({"m-test": m},
                  json.dumps({"machine_id": "m-test", "fault": "clear"}), machine_id=None)
    assert m.fault is None


def test_sensor_topic_layouts() -> None:
    assert sensor_topic(None, None, "machine-01") == "edgesense/sensors/machine-01"
    assert sensor_topic("acme", "lyon", "pump-07") == "es/acme/lyon/pump-07/sensors"
    # a single tenant coordinate is enough to opt in; the other defaults
    assert sensor_topic("acme", None, "m1") == "es/acme/default/m1/sensors"
    assert sensor_topic(None, "lyon", "m1") == "es/default/lyon/m1/sensors"


def test_control_filter_layouts() -> None:
    assert control_filter(None, None) == "edgesense/control/fault"
    assert control_filter("acme", "lyon") == "es/acme/lyon/+/control"


def test_control_machine_id_parsing() -> None:
    assert control_machine_id("es/acme/lyon/pump-07/control") == "pump-07"
    assert control_machine_id("edgesense/control/fault") is None
    assert control_machine_id("es/acme/lyon/control") is None  # too short
    assert control_machine_id("es/a/b/m/sensors") is None  # not a control topic
