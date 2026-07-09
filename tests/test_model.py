"""Model quality gates on synthetic data."""

from __future__ import annotations

import joblib
import numpy as np

from ml.scoring import score_sample
from ml.train import fault_data, normal_data


def test_model_detects_faults_and_spares_normals(small_model_path) -> None:
    pipeline = joblib.load(small_model_path)["pipeline"]
    rng = np.random.default_rng(1)

    fp_rate = float(np.mean(pipeline.predict(normal_data(2_000, rng)) == -1))
    detection = float(np.mean(pipeline.predict(fault_data(2_000, rng)) == -1))

    assert fp_rate < 0.03, f"too many false positives: {fp_rate:.3%}"
    assert detection > 0.40, f"fault detection too weak: {detection:.3%}"


def test_extreme_faults_are_flagged(small_model_path) -> None:
    """Hybrid scoring must catch single-feature outliers the forest misses."""
    bundle = joblib.load(small_model_path)
    extreme = [
        [4.5, 46.0, 14.0],   # bearing fault vibration
        [0.8, 78.0, 12.0],   # overheat (temperature-only outlier)
        [1.3, 45.0, 23.0],   # overload
    ]
    for x in extreme:
        _, is_anomaly, reason = score_sample(bundle, x)
        assert is_anomaly, f"extreme reading not flagged: {x}"
        assert reason in ("model", "limit", "model+limit")


def test_nominal_point_is_normal(small_model_path) -> None:
    bundle = joblib.load(small_model_path)
    _, is_anomaly, reason = score_sample(bundle, [0.8, 45.0, 12.0])
    assert not is_anomaly
    assert reason is None
