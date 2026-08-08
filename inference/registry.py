"""Fetch the champion bundle from the Azure ML / MLflow model registry.

Closes the loop opened in phase 2: ``ml/register_model.py`` writes champions
*into* the registry, and this module reads them back *out* so serving can
follow a registry rollout instead of whatever was baked into the image.

Champion resolution mirrors how registration records it. Azure ML's MLflow
registry does not implement the alias API (``@champion`` 404s), so
``register_model.py`` records the pointer in tags instead:

    registered model tag   champion_version = "<version>"
    model version tag      role             = "champion"

We read the model-level pointer first and fall back to scanning version tags,
so a registry written by either path still resolves.

``mlflow`` is imported lazily: the inference image only installs it when the
registry source is actually used (``pip install -e '.[inference,registry]'``),
and the file-backed default keeps working without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = os.environ.get("EDGESENSE_MODEL_NAME", "edgesense-anomaly")
BUNDLE_FILENAME = "model.joblib"


class RegistryError(RuntimeError):
    """Registry unreachable, misconfigured, or holding no usable champion."""


@dataclass(frozen=True)
class ChampionRef:
    """A resolved champion: registry version + the bundle downloaded locally."""

    path: Path            # local model.joblib ready for joblib.load
    registry_version: str  # MLflow version number, e.g. "4"
    model_name: str


def _client():
    """MlflowClient bound to MLFLOW_TRACKING_URI. Raises RegistryError."""
    # MLflow retries failed HTTP calls with exponential backoff — 7 retries at
    # factor 2 is ~4 minutes before it gives up. A node booting while the
    # registry is unreachable has to fall back to its baked-in bundle in
    # seconds, not minutes (and a container health probe will not wait), so cap
    # both unless the operator has tuned them deliberately.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - needs the extra uninstalled
        raise RegistryError(
            "mlflow is not installed; install the 'registry' extra to serve "
            "from the model registry"
        ) from exc
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RegistryError("MLFLOW_TRACKING_URI is not set")
    try:
        return MlflowClient()
    except Exception as exc:
        raise RegistryError(f"could not create an MLflow client: {exc}") from exc


def resolve_champion_version(client, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Return the registry version number currently tagged champion.

    Prefers the model-level ``champion_version`` pointer that
    ``register_model.py --promote`` writes; falls back to the newest version
    carrying ``role=champion``.
    """
    try:
        model = client.get_registered_model(model_name)
    except Exception as exc:
        raise RegistryError(f"model {model_name!r} not found in registry: {exc}") from exc

    pointer = (getattr(model, "tags", None) or {}).get("champion_version")
    if pointer:
        return str(pointer)

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        raise RegistryError(f"could not list versions of {model_name!r}: {exc}") from exc

    champions = [v for v in versions
                 if (getattr(v, "tags", None) or {}).get("role") == "champion"]
    if not champions:
        raise RegistryError(
            f"{model_name!r} has no champion: neither a champion_version tag nor "
            f"any version tagged role=champion (promote one with "
            f"`register_model.py --promote`)")
    # Version numbers are strings in MLflow; compare numerically where possible.
    return str(max(champions, key=lambda v: int(v.version)).version)


def fetch_champion(model_name: str = DEFAULT_MODEL_NAME,
                   dest_dir: Path | str | None = None) -> ChampionRef:
    """Download the champion bundle and return a reference to the local copy.

    The caller still validates the bundle before serving it — this function is
    only responsible for getting the bytes onto local disk.
    """
    client = _client()
    version = resolve_champion_version(client, model_name)

    try:
        mv = client.get_model_version(model_name, version)
    except Exception as exc:
        raise RegistryError(
            f"could not read {model_name!r} v{version}: {exc}") from exc

    source = getattr(mv, "source", None)
    if not source:
        raise RegistryError(f"{model_name!r} v{version} has no artifact source URI")

    dest = Path(dest_dir) if dest_dir is not None else Path(
        os.environ.get("EDGESENSE_REGISTRY_CACHE", "/tmp/edgesense-registry"))
    dest.mkdir(parents=True, exist_ok=True)

    try:
        import mlflow.artifacts as artifacts
        local = artifacts.download_artifacts(artifact_uri=source, dst_path=str(dest))
    except RegistryError:
        raise
    except Exception as exc:
        raise RegistryError(
            f"could not download artifacts for {model_name!r} v{version}: {exc}") from exc

    bundle = Path(local) / BUNDLE_FILENAME
    if not bundle.exists():
        # register_model.py logs the bundle dir under artifact_path="model";
        # tolerate a flat layout too rather than failing the rollout.
        found = next(Path(local).rglob(BUNDLE_FILENAME), None)
        if found is None:
            raise RegistryError(
                f"{BUNDLE_FILENAME} not found in the artifacts of "
                f"{model_name!r} v{version} (downloaded to {local})")
        bundle = found

    return ChampionRef(path=bundle, registry_version=str(version), model_name=model_name)
