"""Shared fixtures: a small, fast-to-train model bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train import FEATURES, normal_data  # noqa: E402


@pytest.fixture(scope="session")
def small_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(0)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", IsolationForest(n_estimators=100, contamination=0.005, random_state=0)),
    ])
    pipeline.fit(normal_data(5_000, rng))
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "z_guard": 6.0}, path)
    return path
