"""Shared scoring logic: autoencoder reconstruction error + hard z-score guard.

The model is a small autoencoder trained on healthy data only (ml/train.py).
Its anomaly score is the mean squared reconstruction error in scaled feature
space — healthy readings reconstruct well (low score), faults do not (high
score, i.e. higher = more anomalous). A reading is model-flagged when the
error exceeds a threshold calibrated on held-out healthy data.

The z-score guard stays: a model can in principle reconstruct (and thus miss)
drift that resembles a compressed healthy pattern, and industrial practice
combines models with per-feature limit checks. A reading is anomalous if the
model says so OR any feature deviates more than `z_guard` standard deviations
from the training distribution.

Legacy IsolationForest bundles (``ml/train.py --model iforest``, kept as a
comparison baseline) still score through here; their score keeps sklearn's
decision_function semantics (negative = anomalous).
"""

from __future__ import annotations

import numpy as np

DEFAULT_Z_GUARD = 6.0

_ACTIVATIONS = {"tanh": np.tanh}


def ae_reconstruct(bundle: dict, z: np.ndarray) -> np.ndarray:
    """Forward pass of the bundled autoencoder on scaled readings z (n, d).

    The bundle stores raw numpy weights, so serving needs neither torch nor a
    fitted sklearn estimator regardless of which backend trained the model.
    """
    act = _ACTIVATIONS[bundle["activation"]]
    h = z
    last = len(bundle["weights"]) - 1
    for i, (w, b) in enumerate(bundle["weights"]):
        h = h @ w + b
        if i < last:
            h = act(h)
    return h


def reconstruction_errors(bundle: dict, x: "np.ndarray | list") -> np.ndarray:
    """Anomaly scores (mean squared reconstruction error, scaled space) for a batch."""
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    z = (arr - bundle["scaler_mean"]) / bundle["scaler_scale"]
    recon = ae_reconstruct(bundle, z)
    return np.mean((z - recon) ** 2, axis=1)


def score_sample(bundle: dict, x: "np.ndarray | list[float]") -> tuple[float, bool, str | None]:
    """Score one reading. Returns (score, is_anomaly, reason).

    reason is one of None, "model", "limit", "model+limit".
    """
    arr = np.asarray(x, dtype=float).reshape(1, -1)

    if bundle.get("kind", "iforest") == "autoencoder":
        score = float(reconstruction_errors(bundle, arr)[0])
        model_hit = score > bundle["threshold"]
        mean, scale = bundle["scaler_mean"], bundle["scaler_scale"]
    else:  # legacy IsolationForest pipeline bundle
        pipeline = bundle["pipeline"]
        score = float(pipeline.decision_function(arr)[0])
        model_hit = bool(pipeline.predict(arr)[0] == -1)
        scaler = pipeline.named_steps["scaler"]
        mean, scale = scaler.mean_, scaler.scale_

    guard = bundle.get("z_guard", DEFAULT_Z_GUARD)
    max_z = float(np.max(np.abs((arr - mean) / scale)))
    limit_hit = max_z > guard

    reason = {(False, False): None, (True, False): "model",
              (False, True): "limit", (True, True): "model+limit"}[(model_hit, limit_hit)]
    return score, model_hit or limit_hit, reason
