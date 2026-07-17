"""Model manifest, versioning and model-card generation (MLOps phase 1).

Every trained bundle gains a *manifest*: a small JSON-serializable dict that
records what the model is (version, architecture, backend), where it came from
(git commit, training config, training-data descriptor with a content hash)
and how it performed at training time (metrics snapshot). The manifest is

- embedded in the joblib bundle under the ``"manifest"`` key, and
- written as a sidecar ``<bundle>.manifest.json`` next to the bundle, with a
  human-readable ``MODEL_CARD.md`` rendered from it.

Version scheme: ``{YYYYMMDD.HHMMSS}+{git7}`` (UTC) — sortable, unique per
retrain, traceable to a commit. Inside Docker builds ``.git`` is not part of
the context, so the commit falls back to the ``EDGESENSE_GIT_COMMIT`` build
arg and finally to ``"nogit"``.

Backward compatibility: the manifest is additive. ``ml/scoring.py`` ignores
unknown bundle keys, and consumers must treat the manifest as optional
(``bundle.get("manifest")``) so pre-phase-1 bundles keep loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

SCHEMA_VERSION = 1

MODEL_CARD_NAME = "MODEL_CARD.md"


def git_commit() -> str:
    """Short git commit of the training tree, or a graceful fallback."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("EDGESENSE_GIT_COMMIT", "").strip()[:7] or "nogit"


def data_sha256(x: np.ndarray) -> str:
    """Content hash of a training matrix (row-major float64 bytes)."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_manifest(bundle: dict, *, seed: int, epochs: int,
                   training_data: dict, metrics: dict | None = None,
                   created_at: float | None = None) -> dict:
    """Assemble the manifest for a freshly trained autoencoder bundle."""
    ts = time.gmtime(created_at if created_at is not None else time.time())
    commit = git_commit()
    layers = [len(bundle["features"])] + [w.shape[1] for w, _ in bundle["weights"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": f"{time.strftime('%Y%m%d.%H%M%S', ts)}+{commit}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", ts),
        "git_commit": commit,
        "kind": bundle["kind"],
        "backend": bundle.get("backend"),
        "features": list(bundle["features"]),
        "training": {
            "seed": seed,
            "epochs": epochs,
            "architecture": {
                "layers": layers,
                "activation": bundle["activation"],
            },
            "fp_budget": training_data.get("fp_budget"),
            "z_guard": bundle.get("z_guard"),
            "n_train": training_data.get("n_train"),
            "n_cal": training_data.get("n_cal"),
        },
        "training_data": {
            "generator": training_data.get("generator", "synthetic-normal-v1"),
            "params": training_data.get("params", {}),
            "sha256": training_data.get("sha256"),
        },
        "metrics": dict(metrics or {}),
    }


def manifest_path(bundle_path: "Path | str") -> Path:
    """Sidecar JSON path for a bundle: model.joblib -> model.manifest.json."""
    p = Path(bundle_path)
    return p.with_name(p.stem + ".manifest.json")


def render_model_card(manifest: dict) -> str:
    """Human-readable model card generated from the manifest."""
    tr = manifest.get("training", {})
    arch = tr.get("architecture", {})
    td = manifest.get("training_data", {})
    metrics = manifest.get("metrics", {})

    lines = [
        "# EdgeSense AI — model card",
        "",
        f"- **model version**: `{manifest['model_version']}`",
        f"- **created**: {manifest['created_at']} (git `{manifest['git_commit']}`)",
        f"- **kind / backend**: {manifest.get('kind')} / {manifest.get('backend')}",
        f"- **features**: {', '.join(manifest.get('features', []))}",
        "",
        "## Architecture & training",
        "",
        f"- layers: {' → '.join(str(d) for d in arch.get('layers', []))}"
        f" ({arch.get('activation')})",
        f"- seed {tr.get('seed')}, epochs {tr.get('epochs')}",
        f"- false-positive budget: {tr.get('fp_budget')}"
        f" · z-guard: {tr.get('z_guard')}σ",
        f"- training / calibration samples: {tr.get('n_train')} / {tr.get('n_cal')}",
        "",
        "## Training data",
        "",
        f"- generator: `{td.get('generator')}`",
        f"- sha256: `{td.get('sha256')}`",
    ]
    for feat, params in (td.get("params") or {}).items():
        desc = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"- `{feat}`: {desc}")
    lines += ["", "## Metrics snapshot", ""]
    if metrics:
        lines += [f"- {k}: {v}" for k, v in metrics.items()]
    else:
        lines.append("- (none recorded)")
    lines += [
        "",
        "## Intended use & limits",
        "",
        "- Scores one reading at a time via `POST /score`; anomaly = reconstruction",
        "  error above the calibrated threshold OR any feature beyond the z-guard.",
        "- Trained on synthetic healthy data only — see `docs/EVALUATION.md` for the",
        "  offline quality bar and `docs/MLOPS.md` for the promotion gate that",
        "  every new model must pass.",
        "",
        f"*(generated by `ml/manifest.py`, manifest schema v{manifest['schema_version']})*",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_dump_joblib(bundle: dict, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp")
    os.close(fd)
    try:
        joblib.dump(bundle, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_bundle(bundle: dict, path: "Path | str") -> Path:
    """Atomically write the bundle plus (if present) manifest JSON + model card.

    Returns the bundle path. Files are written via temp-file + ``os.replace``
    so a concurrently reloading server never observes a half-written model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_dump_joblib(bundle, path)

    manifest = bundle.get("manifest")
    if manifest:
        _atomic_write_bytes(manifest_path(path),
                            json.dumps(manifest, indent=2).encode())
        _atomic_write_bytes(path.parent / MODEL_CARD_NAME,
                            render_model_card(manifest).encode())
    return path
