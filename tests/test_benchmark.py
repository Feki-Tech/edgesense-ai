"""Hermetic tests for ml/benchmark_public.py (no network, synthetic data).

The real AI4I 2020 download is exercised only by `make benchmark`; here we
feed a small synthetic DataFrame with the same schema through the benchmark
pipeline to keep CI offline and fast.
"""

import numpy as np
import pandas as pd
import pytest

from ml.benchmark_public import FAILURE_MODES, FEATURES, LABEL, run_benchmark, to_markdown


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """AI4I-shaped frame: correlated healthy rows + separable per-mode faults."""
    rng = np.random.default_rng(0)
    n = 1200
    air = rng.normal(300.0, 2.0, n)
    process = air + rng.normal(10.0, 1.0, n)          # correlated with air temp
    speed = rng.normal(1500.0, 100.0, n)
    torque = rng.normal(40.0, 8.0, n)
    wear = rng.uniform(0.0, 200.0, n)
    healthy = pd.DataFrame(
        dict(zip(FEATURES, [air, process, speed, torque, wear])))
    healthy[LABEL] = 0
    for mode in FAILURE_MODES:
        healthy[mode] = 0

    faults = []
    for i, mode in enumerate(FAILURE_MODES):
        rows = healthy.iloc[:20].copy()
        rows[FEATURES[i]] += 40 * healthy[FEATURES[i]].std()  # blatant excursion
        rows[LABEL] = 1
        rows[mode] = 1
        faults.append(rows)
    return pd.concat([healthy, *faults], ignore_index=True)


@pytest.fixture(scope="module")
def results(synthetic_df: pd.DataFrame) -> dict:
    return run_benchmark(synthetic_df, "sklearn", seed=0, epochs=200)


def test_result_schema(results: dict) -> None:
    assert set(results["modes"]) == set(FAILURE_MODES)
    for r in results["modes"].values():
        assert 0.0 <= r["auc"] <= 1.0
        assert 0.0 <= r["model_recall"] <= r["hybrid_recall"] <= 1.0
    assert 0.0 <= results["auc"] <= 1.0
    assert results["healthy"]["n"] > 0


def test_detects_blatant_faults(results: dict) -> None:
    for mode, r in results["modes"].items():
        assert r["hybrid_recall"] == 1.0, f"{mode} missed a 40-sigma excursion"
    assert results["auc"] > 0.99


def test_healthy_fp_bounded(results: dict) -> None:
    assert results["healthy"]["model_fp"] <= 0.03


def test_markdown_report(results: dict) -> None:
    md = to_markdown(results, {"seed": 0, "date": "2026-01-01"})
    assert md.startswith("# EdgeSense AI")
    for mode in FAILURE_MODES:
        assert mode in md
    assert "ROC-AUC" in md
