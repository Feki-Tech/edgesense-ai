"""Train the EdgeSense anomaly model.

Generates synthetic *normal* operating data (same distributions as the
simulator's healthy regime), fits an IsolationForest pipeline and validates
it against synthetic fault data. Saves the model to ml/model/model.joblib.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["vibration", "temperature", "current"]
MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"


def normal_data(n: int, rng: np.random.Generator) -> np.ndarray:
    vib = 0.8 + rng.normal(0, 0.15, n)
    temp = 45.0 + rng.normal(0, 1.2, n)
    cur = 12.0 + rng.normal(0, 0.6, n)
    return np.column_stack([np.clip(vib, 0, None), temp, np.clip(cur, 0, None)])


def fault_data(n: int, rng: np.random.Generator) -> np.ndarray:
    third = n // 3
    # bearing fault: high vibration, slightly high current
    bearing = np.column_stack([
        (0.8 + rng.normal(0, 0.15, third)) * rng.uniform(3.0, 5.0, third),
        45.0 + rng.normal(0, 1.2, third),
        (12.0 + rng.normal(0, 0.6, third)) * rng.uniform(1.1, 1.25, third),
    ])
    # overheat: temperature 15-30 C above normal
    overheat = np.column_stack([
        0.8 + rng.normal(0, 0.15, third),
        45.0 + rng.normal(0, 1.2, third) + rng.uniform(15, 30, third),
        12.0 + rng.normal(0, 0.6, third),
    ])
    # overload: high current, elevated vibration
    rest = n - 2 * third
    overload = np.column_stack([
        (0.8 + rng.normal(0, 0.15, rest)) * rng.uniform(1.4, 1.8, rest),
        45.0 + rng.normal(0, 1.2, rest),
        (12.0 + rng.normal(0, 0.6, rest)) * rng.uniform(1.6, 2.0, rest),
    ])
    return np.vstack([bearing, overheat, overload])


def main() -> None:
    rng = np.random.default_rng(42)
    x_train = normal_data(20_000, rng)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", IsolationForest(n_estimators=200, contamination=0.005, random_state=42)),
    ])
    model.fit(x_train)

    # validation
    x_normal = normal_data(2_000, rng)
    x_fault = fault_data(2_000, rng)
    fp = float(np.mean(model.predict(x_normal) == -1))
    tp = float(np.mean(model.predict(x_fault) == -1))
    print(f"validation: false-positive rate on normal = {fp:.3%}")
    print(f"validation: detection rate on faults      = {tp:.3%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": FEATURES}, MODEL_PATH)
    print(f"model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
