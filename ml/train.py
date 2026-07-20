"""Train the EdgeSense anomaly model.

Generates synthetic *normal* operating data (same distributions as the
simulator's healthy regime) and fits a small autoencoder (3 -> 16 -> 2 -> 16
-> 3, tanh) to reconstruct it. Healthy readings reconstruct well; faults do
not, so the mean squared reconstruction error in scaled feature space is the
anomaly score (higher = more anomalous). The alarm threshold is calibrated on
held-out healthy data at the (1 - FP_BUDGET) quantile (~0.5% false-positive
budget, mirroring the old IsolationForest contamination).

Two interchangeable training backends emit the *same* bundle format — raw
numpy weights, so inference (ml/scoring.py) needs neither torch nor a fitted
sklearn estimator:

    python ml/train.py                        # sklearn MLPRegressor (default)
    python ml/train.py --backend torch        # PyTorch, uses CUDA if available
    python ml/train.py --model iforest        # legacy IsolationForest baseline

Saves the bundle to ml/model/model.joblib (or --out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.manifest import build_manifest, data_sha256, save_bundle  # noqa: E402
from ml.scoring import DEFAULT_Z_GUARD, reconstruction_errors  # noqa: E402

FEATURES = ["vibration", "temperature", "current"]
MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"

HIDDEN_LAYERS = (16, 2, 16)  # 2-unit bottleneck for the 3 input features
ACTIVATION = "tanh"          # saturates outside the healthy region -> big fault errors
FP_BUDGET = 0.005            # target false-positive rate on healthy data

# healthy-regime distributions of normal_data() — recorded in the manifest so a
# bundle is traceable to the exact generative process that trained it
GENERATOR_PARAMS = {
    "vibration": {"mean": 0.8, "std": 0.15, "clip_min": 0.0},
    "temperature": {"mean": 45.0, "std": 1.2},
    "current": {"mean": 12.0, "std": 0.6, "clip_min": 0.0},
}


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


def _fit_sklearn(z_train: np.ndarray, seed: int, epochs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit the autoencoder with sklearn's MLPRegressor (no extra dependencies)."""
    from sklearn.neural_network import MLPRegressor

    mlp = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation=ACTIVATION,
                       solver="adam", learning_rate_init=5e-3, max_iter=epochs,
                       n_iter_no_change=20, tol=1e-5, random_state=seed)
    mlp.fit(z_train, z_train)
    return [(w.astype(np.float64), b.astype(np.float64))
            for w, b in zip(mlp.coefs_, mlp.intercepts_)]


def _fit_torch(z_train: np.ndarray, seed: int, epochs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit the autoencoder with PyTorch (optional; CUDA if available)."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "the torch backend needs PyTorch: "
            "pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dims = [z_train.shape[1], *HIDDEN_LAYERS, z_train.shape[1]]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    model = nn.Sequential(*layers).to(device)

    data = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    batch = 512

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=gen).to(device)
        for start in range(0, len(data), batch):
            xb = data[perm[start:start + batch]]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        # torch Linear stores weight as (out, in); the bundle uses (in, out)
        return [(m.weight.T.cpu().numpy().astype(np.float64),
                 m.bias.cpu().numpy().astype(np.float64))
                for m in model if isinstance(m, nn.Linear)]


_BACKENDS = {"sklearn": (_fit_sklearn, 500), "torch": (_fit_torch, 300)}


def train_autoencoder(backend: str = "sklearn", *, seed: int = 42, n_train: int = 20_000,
                      n_cal: int = 20_000, epochs: int | None = None) -> dict:
    """Train the autoencoder and calibrate its alarm threshold. Returns the bundle."""
    rng = np.random.default_rng(seed)
    x_train, x_cal = normal_data(n_train, rng), normal_data(n_cal, rng)
    fit_epochs = epochs or _BACKENDS[backend][1]
    bundle = build_bundle(x_train, x_cal, FEATURES, backend, seed=seed, epochs=fit_epochs)
    bundle["manifest"] = build_manifest(
        bundle, seed=seed, epochs=fit_epochs,
        training_data={
            "generator": "synthetic-normal-v1",
            "params": GENERATOR_PARAMS,
            "sha256": data_sha256(x_train),
            "fp_budget": FP_BUDGET,
            "n_train": n_train,
            "n_cal": n_cal,
        },
        metrics={"threshold": bundle["threshold"]},
    )
    return bundle


def build_bundle(x_train: np.ndarray, x_cal: np.ndarray, features: list[str],
                 backend: str = "sklearn", *, seed: int = 42,
                 epochs: int | None = None) -> dict:
    """Fit an autoencoder on healthy readings and calibrate the alarm threshold.

    Backend-agnostic: also used by ml/benchmark_public.py to train the same
    architecture on real public datasets with a different feature set.
    """
    fit, default_epochs = _BACKENDS[backend]

    x_train = np.asarray(x_train, dtype=float)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)  # constant features stay centered
    weights = fit((x_train - mean) / scale, seed, epochs or default_epochs)

    bundle = {
        "kind": "autoencoder",
        "backend": backend,
        "features": list(features),
        "activation": ACTIVATION,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "weights": weights,
        "threshold": 0.0,
        "z_guard": DEFAULT_Z_GUARD,
    }
    # calibrate on held-out healthy data with the exact serving arithmetic
    errors = reconstruction_errors(bundle, np.asarray(x_cal, dtype=float))
    bundle["threshold"] = float(np.quantile(errors, 1.0 - FP_BUDGET))
    return bundle


def train_iforest(seed: int = 42, n_train: int = 20_000) -> dict:
    """Legacy IsolationForest baseline, kept for offline comparison."""
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", IsolationForest(n_estimators=200, contamination=FP_BUDGET,
                                   random_state=seed)),
    ])
    pipeline.fit(normal_data(n_train, rng))
    return {"kind": "iforest", "pipeline": pipeline, "features": FEATURES,
            "z_guard": DEFAULT_Z_GUARD}


def validate(bundle: dict, rng: np.random.Generator) -> tuple[float, float]:
    """(false-positive rate on fresh normals, detection rate on synthetic faults)."""
    x_normal, x_fault = normal_data(2_000, rng), fault_data(2_000, rng)
    if bundle["kind"] == "autoencoder":
        thr = bundle["threshold"]
        fp = float(np.mean(reconstruction_errors(bundle, x_normal) > thr))
        tp = float(np.mean(reconstruction_errors(bundle, x_fault) > thr))
    else:
        pipeline = bundle["pipeline"]
        fp = float(np.mean(pipeline.predict(x_normal) == -1))
        tp = float(np.mean(pipeline.predict(x_fault) == -1))
    return fp, tp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("autoencoder", "iforest"), default="autoencoder")
    ap.add_argument("--backend", choices=tuple(_BACKENDS), default="sklearn",
                    help="autoencoder training backend (default: sklearn)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (default: 500 sklearn / 300 torch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(MODEL_PATH))
    args = ap.parse_args()

    if args.model == "iforest":
        bundle = train_iforest(seed=args.seed)
        print("trained legacy IsolationForest baseline")
    else:
        bundle = train_autoencoder(args.backend, seed=args.seed, epochs=args.epochs)
        print(f"trained autoencoder ({args.backend} backend, "
              f"layers {(len(FEATURES), *HIDDEN_LAYERS, len(FEATURES))}, "
              f"threshold {bundle['threshold']:.4f})")

    fp, tp = validate(bundle, np.random.default_rng(args.seed + 1))
    print(f"validation: false-positive rate on normal = {fp:.3%}")
    print(f"validation: detection rate on faults      = {tp:.3%}")

    if "manifest" in bundle:
        bundle["manifest"]["metrics"].update(
            {"val_fp_rate": round(fp, 5), "val_detection_rate": round(tp, 5)})
        print(f"model version: {bundle['manifest']['model_version']}")

    out = save_bundle(bundle, args.out)
    print(f"model saved -> {out}"
          + (" (+ manifest + model card)" if "manifest" in bundle else ""))


if __name__ == "__main__":
    main()
