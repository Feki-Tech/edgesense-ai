"""EdgeSense inference sidecar.

Serves the trained model over HTTP for the Go edge agent.

    POST /score   {"vibration": .., "temperature": .., "current": ..}
    -> {"score": -0.12, "is_anomaly": true, "reason": "model"}

Score is the IsolationForest decision function (negative = anomalous);
is_anomaly combines the model verdict with hard z-score limits
(see ml/scoring.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.scoring import score_sample  # noqa: E402

MODEL_PATH = Path(os.environ.get(
    "EDGESENSE_MODEL",
    Path(__file__).resolve().parent.parent / "ml" / "model" / "model.joblib",
))

app = FastAPI(title="EdgeSense Inference")
_bundle = joblib.load(MODEL_PATH)
_features = _bundle["features"]


class Reading(BaseModel):
    vibration: float
    temperature: float
    current: float


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": str(MODEL_PATH), "features": _features}


@app.post("/score")
def score(reading: Reading) -> dict:
    s, anomaly, reason = score_sample(_bundle, [getattr(reading, f) for f in _features])
    return {"score": round(s, 5), "is_anomaly": anomaly, "reason": reason}
