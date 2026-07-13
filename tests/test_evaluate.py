"""Evaluation harness tests: the shipped detector setup must catch every
simulated episode quickly and stay quiet on healthy data."""

from __future__ import annotations

import joblib

from ml.evaluate import evaluate, to_markdown
from simulator.simulate import FAULT_TYPES


def test_every_episode_detected_fast(small_model_path) -> None:
    bundle = joblib.load(small_model_path)
    results = evaluate(bundle, episodes=4, ticks=20, healthy=2_000, seed=1)

    assert results["healthy"]["fp_rate"] < 0.05
    for fault in FAULT_TYPES:
        r = results["faults"][fault]
        assert r["episode_rate"] == 1.0, f"{fault}: missed episodes"
        assert r["median_latency_readings"] <= 5, f"{fault}: too slow"
        assert r["reading_recall"] > 0.5


def test_markdown_report_structure(small_model_path) -> None:
    bundle = joblib.load(small_model_path)
    results = evaluate(bundle, episodes=2, ticks=10, healthy=500, seed=2)
    md = to_markdown(results, {"model": "test.joblib", "episodes": 2,
                               "ticks": 10, "seed": 2, "date": "2026-01-01"})
    assert "| Fault |" in md
    for fault in FAULT_TYPES:
        assert f"| {fault} |" in md
    assert "false positives" in md
