"""Shared fixtures: small, fast-to-train model bundles."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train import FEATURES, normal_data, train_autoencoder  # noqa: E402


@pytest.fixture(scope="session")
def small_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Autoencoder bundle trained exactly like production, on less data."""
    bundle = train_autoencoder("sklearn", seed=0, n_train=5_000, n_cal=3_000, epochs=300)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture(scope="session")
def iforest_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Legacy IsolationForest bundle (comparison baseline; exercises dispatch)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", IsolationForest(n_estimators=100, contamination=0.005, random_state=0)),
    ])
    pipeline.fit(normal_data(5_000, rng))
    path = tmp_path_factory.mktemp("model-iforest") / "model.joblib"
    joblib.dump({"kind": "iforest", "pipeline": pipeline, "features": FEATURES,
                 "z_guard": 6.0}, path)
    return path
