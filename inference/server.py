"""EdgeSense inference sidecar.

Serves the trained model over HTTP for the Go edge agent.

    POST /score   {"vibration": .., "temperature": .., "current": ..}
    -> {"score": 38.2, "is_anomaly": true, "reason": "model"}

Score is the autoencoder's mean squared reconstruction error in scaled
feature space (higher = more anomalous); is_anomaly combines the model
verdict (error above the calibrated threshold) with hard z-score limits
(see ml/scoring.py).

MLOps phase 1 additions (the /score contract is unchanged):

- GET  /healthz  also reports model_version + created_at from the manifest
- GET  /metrics  Prometheus metrics: scored counter, score histogram, and
                 per-feature drift gauges (z-shift + PSI vs training stats)
- POST /reload   atomically re-loads the bundle from disk (also on SIGHUP
                 where the platform has it); the old model keeps serving if
                 the new file is missing or invalid

Shadow scoring (§2.5 of docs/MLOPS.md — online champion/challenger evidence):

- POST /shadow/load    load the candidate bundle (EDGESENSE_SHADOW_MODEL,
                       default ml/model/candidate/model.joblib) as a shadow
- GET  /shadow         live agreement report (verdicts, score MAE/bias)
- POST /shadow/unload  stop shadowing and return the final report
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import metrics  # noqa: E402
from inference.drift import DEFAULT_WINDOW, DriftTracker  # noqa: E402
from inference.shadow import ShadowTracker  # noqa: E402
from ml.scoring import _ACTIVATIONS, score_sample  # noqa: E402

MODEL_PATH = Path(os.environ.get(
    "EDGESENSE_MODEL",
    Path(__file__).resolve().parent.parent / "ml" / "model" / "model.joblib",
))
SHADOW_PATH = Path(os.environ.get(
    "EDGESENSE_SHADOW_MODEL",
    Path(__file__).resolve().parent.parent / "ml" / "model" / "candidate" / "model.joblib",
))
DRIFT_WINDOW = int(os.environ.get("EDGESENSE_DRIFT_WINDOW", DEFAULT_WINDOW))


class _ModelState:
    """Immutable snapshot of a loaded bundle; swapped atomically on reload."""

    def __init__(self, bundle: dict, path: Path) -> None:
        self.bundle = bundle
        self.path = path
        self.features: list[str] = list(bundle["features"])
        manifest = bundle.get("manifest") or {}
        self.model_version: str = manifest.get("model_version", "unknown")
        self.created_at: str | None = manifest.get("created_at")

    def drift_stats(self) -> "tuple[np.ndarray, np.ndarray]":
        if self.bundle.get("kind", "iforest") == "autoencoder":
            return self.bundle["scaler_mean"], self.bundle["scaler_scale"]
        scaler = self.bundle["pipeline"].named_steps["scaler"]
        return scaler.mean_, scaler.scale_


def _validate_bundle(bundle: object) -> dict:
    """Sanity-check a candidate bundle before it may serve. Raises ValueError."""
    if not isinstance(bundle, dict):
        raise ValueError("bundle is not a dict")
    features = bundle.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("bundle has no feature list")

    kind = bundle.get("kind", "iforest")
    if kind == "autoencoder":
        for key in ("scaler_mean", "scaler_scale", "weights", "threshold", "activation"):
            if key not in bundle:
                raise ValueError(f"autoencoder bundle missing {key!r}")
        if bundle["activation"] not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {bundle['activation']!r}")
        n = len(features)
        mean = np.asarray(bundle["scaler_mean"], dtype=float)
        scale = np.asarray(bundle["scaler_scale"], dtype=float)
        if mean.shape != (n,) or scale.shape != (n,):
            raise ValueError("scaler shape does not match the feature list")
        dim = n
        for i, (w, b) in enumerate(bundle["weights"]):
            if w.shape[0] != dim or w.shape[1] != b.shape[0]:
                raise ValueError(f"weight chain broken at layer {i}")
            dim = w.shape[1]
        if dim != n:
            raise ValueError("autoencoder output dimension != feature count")
        if not np.isfinite(float(bundle["threshold"])):
            raise ValueError("threshold is not finite")
    elif kind == "iforest":
        if "pipeline" not in bundle:
            raise ValueError("iforest bundle missing 'pipeline'")
    else:
        raise ValueError(f"unknown bundle kind {kind!r}")

    # smoke-score a nominal reading with the exact serving arithmetic
    score, _, _ = score_sample(bundle, [0.0] * len(features))
    if not np.isfinite(score):
        raise ValueError("bundle produced a non-finite score")
    return bundle


def _load_state(path: Path) -> _ModelState:
    return _ModelState(_validate_bundle(joblib.load(path)), path)


app = FastAPI(title="EdgeSense Inference")
app.mount("/metrics", metrics.metrics_app())

_state = _load_state(MODEL_PATH)
_state_lock = threading.Lock()  # serializes reloads and shadow swaps, not scoring
_drift = DriftTracker(_state.features, *_state.drift_stats(), window=DRIFT_WINDOW)
_shadow: "tuple[_ModelState, ShadowTracker] | None" = None  # swapped as one reference

metrics.DRIFT_ZSHIFT.clear()
metrics.DRIFT_PSI.clear()
metrics.set_model_info(_state.model_version, _state.bundle.get("kind", "iforest"),
                       str(_state.bundle.get("backend")))
metrics.set_shadow_info(None)  # a module (re)load never inherits a stale shadow


def _swap_state(new_state: _ModelState) -> None:
    global _state, _shadow
    _state = new_state  # atomic reference swap; in-flight requests keep the old one
    _drift.reset(*new_state.drift_stats())
    metrics.DRIFT_ZSHIFT.clear()
    metrics.DRIFT_PSI.clear()
    metrics.set_model_info(new_state.model_version,
                           new_state.bundle.get("kind", "iforest"),
                           str(new_state.bundle.get("backend")))
    # Shadow evidence is per-champion: a new champion starts a fresh tracker.
    shadow = _shadow
    if shadow is not None:
        _shadow = (shadow[0], ShadowTracker(shadow[0].model_version,
                                            new_state.model_version))


class Reading(BaseModel):
    vibration: float
    temperature: float
    current: float


@app.get("/healthz")
def healthz() -> dict:
    state = _state
    return {"status": "ok", "model": str(state.path), "features": state.features,
            "model_kind": state.bundle.get("kind", "iforest"),
            "model_version": state.model_version, "created_at": state.created_at}


@app.post("/score")
def score(reading: Reading) -> dict:
    state = _state  # one snapshot per request; never sees a half-swapped bundle
    x = [getattr(reading, f) for f in state.features]
    s, anomaly, reason = score_sample(state.bundle, x)

    _drift.observe(x)
    metrics.SCORED.inc()
    metrics.SCORE.observe(s)
    if anomaly:
        metrics.ANOMALIES.labels(reason=reason).inc()
    metrics.DRIFT_WINDOW.set(_drift.size)
    for feature, sig in _drift.signals().items():
        metrics.DRIFT_ZSHIFT.labels(feature=feature).set(sig["zshift"])
        metrics.DRIFT_PSI.labels(feature=feature).set(sig["psi"])

    shadow = _shadow
    if shadow is not None:
        shadow_state, tracker = shadow
        try:
            xs = [getattr(reading, f) for f in shadow_state.features]
            shadow_score, shadow_anomaly, _ = score_sample(shadow_state.bundle, xs)
        except Exception:
            tracker.error()
            metrics.SHADOW_ERRORS.inc()
        else:
            tracker.observe(s, anomaly, shadow_score, shadow_anomaly)
            metrics.SHADOW_SCORED.inc()
            metrics.SHADOW_SCORE_DIFF.observe(abs(shadow_score - s))
            if shadow_anomaly != anomaly:
                metrics.SHADOW_DISAGREEMENTS.labels(
                    kind="shadow_only" if shadow_anomaly else "champion_only").inc()

    return {"score": round(s, 5), "is_anomaly": anomaly, "reason": reason}


@app.post("/reload")
def reload_model() -> dict:
    """Re-load the bundle from disk and swap it in atomically.

    Returns the old and new model versions; on any load/validation error the
    current model keeps serving and the request fails with 400.
    """
    with _state_lock:
        old = _state
        try:
            new_state = _load_state(MODEL_PATH)
        except Exception as exc:
            metrics.RELOADS.labels(result="rejected").inc()
            raise HTTPException(
                status_code=400,
                detail=f"reload rejected, keeping {old.model_version}: {exc}",
            ) from exc
        _swap_state(new_state)
    metrics.RELOADS.labels(result="ok").inc()
    return {"status": "reloaded", "old_version": old.model_version,
            "new_version": new_state.model_version, "model": str(new_state.path)}


@app.post("/shadow/load")
def shadow_load() -> dict:
    """Load the candidate bundle as a shadow and start a fresh agreement tracker.

    The shadow never answers /score; it only scores the same readings in the
    background. Loading a new shadow (or re-loading) resets the evidence.
    """
    global _shadow
    with _state_lock:
        try:
            shadow_state = _load_state(SHADOW_PATH)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"shadow load rejected: {exc}") from exc
        tracker = ShadowTracker(shadow_state.model_version, _state.model_version)
        _shadow = (shadow_state, tracker)
    metrics.set_shadow_info(shadow_state.model_version)
    return {"status": "shadowing", "model": str(SHADOW_PATH),
            "shadow_version": tracker.shadow_version,
            "champion_version": tracker.champion_version}


@app.get("/shadow")
def shadow_status() -> dict:
    shadow = _shadow
    if shadow is None:
        return {"active": False}
    return {"active": True, "model": str(shadow[0].path), "report": shadow[1].report()}


@app.post("/shadow/unload")
def shadow_unload() -> dict:
    global _shadow
    with _state_lock:
        shadow = _shadow
        if shadow is None:
            raise HTTPException(status_code=404, detail="no shadow loaded")
        _shadow = None
    metrics.set_shadow_info(None)
    return {"status": "unloaded", "report": shadow[1].report()}


if hasattr(signal, "SIGHUP"):  # pragma: no cover - POSIX only
    def _on_sighup(_signum, _frame) -> None:
        try:
            info = reload_model()
            print(f"SIGHUP: reloaded model -> {info['new_version']}")
        except HTTPException as exc:
            print(f"SIGHUP: {exc.detail}", file=sys.stderr)

    try:
        signal.signal(signal.SIGHUP, _on_sighup)
    except ValueError:
        pass  # not in the main thread (e.g. under some test runners)
