"""EdgeSense inference sidecar.

Serves the trained IsolationForest over HTTP for the Go edge agent.

    POST /score   {"vibration": .., "temperature": .., "current": ..}
    -> {"score": -0.12, "is_anomaly": true}

Score is the IsolationForest decision function: negative means anomalous.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get(
    "EDGESENSE_MODEL",
    Path(__file__).resolve().parent.parent / "ml" / "model" / "model.joblib",
))

app = FastAPI(title="EdgeSense Inference")
_bundle = joblib.load(MODEL_PATH)
_pipeline = _bundle["pipeline"]
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
    x = np.array([[getattr(reading, f) for f in _features]])
    s = float(_pipeline.decision_function(x)[0])
    return {"score": round(s, 5), "is_anomaly": bool(_pipeline.predict(x)[0] == -1)}
