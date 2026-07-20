"""Model quality gates on synthetic data."""

from __future__ import annotations

import joblib
import numpy as np

from ml.scoring import reconstruction_errors, score_sample
from ml.train import FEATURES, HIDDEN_LAYERS, fault_data, normal_data


def test_bundle_schema(small_model_path) -> None:
    bundle = joblib.load(small_model_path)
    assert bundle["kind"] == "autoencoder"
    assert bundle["backend"] == "sklearn"
    assert bundle["features"] == FEATURES
    assert bundle["threshold"] > 0
    dims = (len(FEATURES), *HIDDEN_LAYERS, len(FEATURES))
    assert [w.shape for w, _ in bundle["weights"]] == \
        [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def test_model_detects_faults_and_spares_normals(small_model_path) -> None:
    bundle = joblib.load(small_model_path)
    rng = np.random.default_rng(1)
    thr = bundle["threshold"]

    fp_rate = float(np.mean(reconstruction_errors(bundle, normal_data(2_000, rng)) > thr))
    detection = float(np.mean(reconstruction_errors(bundle, fault_data(2_000, rng)) > thr))

    assert fp_rate < 0.03, f"too many false positives: {fp_rate:.3%}"
    assert detection > 0.40, f"fault detection too weak: {detection:.3%}"


def test_reconstruction_error_separates_faults(small_model_path) -> None:
    """Higher score = more anomalous: fault errors must dwarf healthy errors."""
    bundle = joblib.load(small_model_path)
    rng = np.random.default_rng(2)
    healthy = float(np.median(reconstruction_errors(bundle, normal_data(1_000, rng))))
    faulty = float(np.median(reconstruction_errors(bundle, fault_data(1_000, rng))))
    assert faulty > 5 * healthy, f"weak separation: healthy {healthy:.4f} vs fault {faulty:.4f}"


def test_extreme_faults_are_flagged(small_model_path) -> None:
    """Hybrid scoring must catch single-feature outliers whatever the model does."""
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


def test_legacy_iforest_bundle_still_scores(iforest_model_path) -> None:
    """Old-style bundles keep working (and keep decision_function semantics)."""
    bundle = joblib.load(iforest_model_path)

    score, is_anomaly, reason = score_sample(bundle, [0.8, 45.0, 12.0])
    assert not is_anomaly
    assert reason is None
    assert score > 0  # positive = normal for IsolationForest

    _, is_anomaly, reason = score_sample(bundle, [4.5, 46.0, 14.0])
    assert is_anomaly
    assert reason in ("model", "limit", "model+limit")
