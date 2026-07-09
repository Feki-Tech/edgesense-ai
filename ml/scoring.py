"""Shared scoring logic: IsolationForest score + hard z-score guard.

IsolationForest is weak on single-feature outliers (e.g. pure overheat:
temperature alone at +27 sigma can pass unflagged because random splits on the
other, perfectly-normal features mask it). Industrial practice: combine the
model with per-feature limit checks. A reading is anomalous if the model says
so OR any feature deviates more than `z_guard` standard deviations from the
training distribution.
"""

from __future__ import annotations

import numpy as np

DEFAULT_Z_GUARD = 6.0


def score_sample(bundle: dict, x: "np.ndarray | list[float]") -> tuple[float, bool, str | None]:
    """Score one reading. Returns (score, is_anomaly, reason).

    reason is one of None, "model", "limit", "model+limit".
    """
    pipeline = bundle["pipeline"]
    guard = bundle.get("z_guard", DEFAULT_Z_GUARD)

    arr = np.asarray(x, dtype=float).reshape(1, -1)
    score = float(pipeline.decision_function(arr)[0])
    model_hit = bool(pipeline.predict(arr)[0] == -1)

    scaler = pipeline.named_steps["scaler"]
    max_z = float(np.max(np.abs((arr - scaler.mean_) / scaler.scale_)))
    limit_hit = max_z > guard

    reason = {(False, False): None, (True, False): "model",
              (False, True): "limit", (True, True): "model+limit"}[(model_hit, limit_hit)]
    return score, model_hit or limit_hit, reason
