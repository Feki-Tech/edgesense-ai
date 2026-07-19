"""Azure ML managed-endpoint scoring script for the EdgeSense bundle.

Loads the registered bundle (model.joblib) and applies the same hybrid rule
as ml/scoring.py in the main repo: anomaly if reconstruction error exceeds
the calibrated threshold OR any feature deviates more than z_guard sigma
from the training distribution.

NOTE: keep this in sync with edgesense-ai/ml/scoring.py — or better, package
that module and import it here. The bundle layout expected:
    model/model.joblib  with keys: scaler stats, weights, threshold, z_guard
Adjust the key names below to match your actual bundle schema.
"""
from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np

FEATURES = ["vibration", "temperature", "current"]
_bundle = None


def init():
    global _bundle
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    # Registered artifacts land under <AZUREML_MODEL_DIR>/model/
    path = None
    for root, _dirs, files in os.walk(model_dir):
        if "model.joblib" in files:
            path = os.path.join(root, "model.joblib")
            break
    if path is None:
        raise FileNotFoundError(f"model.joblib not found under {model_dir}")
    _bundle = joblib.load(path)
    logging.info("EdgeSense bundle loaded from %s", path)


def _score_one(reading: dict) -> dict:
    x = np.array([[float(reading[f]) for f in FEATURES]])

    # ---- adjust these keys to your bundle schema (see ml/scoring.py) ----
    mean = np.asarray(_bundle["scaler_mean"])
    std = np.asarray(_bundle["scaler_std"])
    threshold = float(_bundle["threshold"])
    z_guard = float(_bundle.get("z_guard", 6.0))

    xs = (x - mean) / std

    # Forward pass through the autoencoder stored as raw numpy weights.
    h = xs
    for w, b in _bundle["layers"]:
        h = np.tanh(h @ np.asarray(w) + np.asarray(b))
    recon_err = float(np.mean((h - xs) ** 2))

    z_max = float(np.max(np.abs(xs)))
    model_hit = recon_err > threshold
    limit_hit = z_max > z_guard
    reason = ("model+limit" if model_hit and limit_hit
              else "model" if model_hit
              else "limit" if limit_hit
              else "none")
    return {
        "anomaly": bool(model_hit or limit_hit),
        "score": recon_err,
        "threshold": threshold,
        "reason": reason,
    }


def run(raw_data):
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    readings = payload if isinstance(payload, list) else [payload]
    return [_score_one(r) for r in readings]
