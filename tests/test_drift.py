"""Serving-side drift detection: tracker math and /metrics exposure."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference.drift import DriftTracker, expected_bin_probs, psi
from ml.train import GENERATOR_PARAMS, normal_data


def _tracker(window: int = 400) -> DriftTracker:
    means = np.array([p["mean"] for p in GENERATOR_PARAMS.values()])
    stds = np.array([p["std"] for p in GENERATOR_PARAMS.values()])
    return DriftTracker(list(GENERATOR_PARAMS), means, stds, window=window)


def test_expected_bin_probs_sum_to_one() -> None:
    assert float(np.sum(expected_bin_probs())) == pytest.approx(1.0)


def test_psi_near_zero_for_matching_distribution() -> None:
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5_000)) < 0.05


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    assert psi(rng.normal(2.0, 1, 5_000)) > 0.25


def test_healthy_stream_keeps_drift_low() -> None:
    tracker = _tracker()
    rng = np.random.default_rng(3)
    for row in normal_data(400, rng):
        tracker.observe(row)

    signals = tracker.signals()
    assert set(signals) == set(GENERATOR_PARAMS)
    for sig in signals.values():
        assert abs(sig["zshift"]) < 0.2
        assert sig["psi"] < 0.1


def test_drifted_stream_raises_the_signals() -> None:
    tracker = _tracker()
    rng = np.random.default_rng(3)
    drifted = normal_data(400, rng)
    drifted[:, 1] += 3.0  # temperature runs ~2.5σ hot — sensor or process drift

    for row in drifted:
        tracker.observe(row)

    sig = tracker.signals()["temperature"]
    assert sig["zshift"] > 1.0
    assert sig["psi"] > 0.25
    # the other features stay quiet
    assert abs(tracker.signals()["vibration"]["zshift"]) < 0.2


def test_no_signal_below_min_samples() -> None:
    tracker = _tracker()
    for _ in range(10):
        tracker.observe([0.8, 45.0, 12.0])
    assert tracker.signals() == {}
    assert tracker.size == 10


def test_reset_clears_the_window() -> None:
    tracker = _tracker()
    for _ in range(60):
        tracker.observe([0.8, 45.0, 12.0])
    assert tracker.signals()
    tracker.reset()
    assert tracker.size == 0
    assert tracker.signals() == {}


def test_window_wraps_and_forgets_old_readings() -> None:
    tracker = _tracker(window=100)
    for _ in range(100):
        tracker.observe([0.8, 45.0, 12.0])          # healthy prefix ...
    for _ in range(100):
        tracker.observe([0.8, 50.0, 12.0])          # ... fully displaced by hot data
    assert tracker.size == 100
    assert tracker.signals()["temperature"]["zshift"] > 3.0


@pytest.fixture()
def client(small_model_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("EDGESENSE_MODEL", str(small_model_path))
    monkeypatch.setenv("EDGESENSE_DRIFT_WINDOW", "200")
    import inference.server as server
    importlib.reload(server)
    return TestClient(server.app)


def test_metrics_endpoint_exposes_drift_gauges(client: TestClient) -> None:
    rng = np.random.default_rng(5)
    for vib, temp, cur in normal_data(80, rng):
        client.post("/score", json={"vibration": vib, "temperature": temp,
                                    "current": cur})

    text = client.get("/metrics").text
    assert "edgesense_model_scored_total" in text
    assert 'edgesense_model_drift_psi{feature="temperature"}' in text
    assert 'edgesense_model_drift_zshift{feature="vibration"}' in text
    assert "edgesense_model_score_bucket" in text
    assert "edgesense_model_info{" in text


def test_scoring_drifted_stream_moves_the_gauge(client: TestClient) -> None:
    import inference.server as server

    rng = np.random.default_rng(6)
    for vib, temp, cur in normal_data(120, rng):
        client.post("/score", json={"vibration": vib, "temperature": temp + 4.0,
                                    "current": cur})

    sig = server._drift.signals()["temperature"]
    assert sig["zshift"] > 1.0
    assert sig["psi"] > 0.25
