"""Registry-driven serving: champion resolution, download, and rollout.

These run against a **real MLflow registry** — a local sqlite-backed tracking
store, which is the lightest backend that implements registered models — and
register through the real ``ml/register_model.py``. So the round trip under
test is the production one: promote writes the champion, ``inference.registry``
resolves and downloads it, and the sidecar validates and serves it.

Azure ML's registry differs in one way that matters (its alias API 404s, so
``register_model.py`` records the champion pointer in tags instead); the tag
path exercised here is the one both backends share.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import time
import uuid
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from ml.manifest import save_bundle

mlflow = pytest.importorskip("mlflow", reason="registry extra not installed")
from mlflow.tracking import MlflowClient  # noqa: E402


@pytest.fixture(scope="session")
def _registry_store(tmp_path_factory):
    """One sqlite tracking store for the whole session.

    Creating it runs MLflow's schema migration, which costs far more than the
    tests themselves — so it is built once and tests isolate from each other by
    registering under a unique model name instead of a fresh database.
    """
    root = tmp_path_factory.mktemp("mlflow-registry")
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}", root


@pytest.fixture()
def local_registry(_registry_store, tmp_path, monkeypatch):
    """Point MLflow at the shared store; yield a model name unique to this test."""
    uri, root = _registry_store
    # Artifacts land under ./mlruns relative to cwd — keep that beside the db so
    # the absolute paths recorded at experiment creation stay valid all session.
    monkeypatch.chdir(root)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("EDGESENSE_REGISTRY_CACHE", str(tmp_path / "cache"))
    mlflow.set_tracking_uri(uri)
    yield f"edgesense-anomaly-{uuid.uuid4().hex[:8]}"
    mlflow.set_tracking_uri(None)


@pytest.fixture()
def bundle_dir(small_model_path, tmp_path):
    """Build a bundle directory shaped like ml/model/ (joblib + manifest)."""
    def _make(version: str) -> Path:
        out = tmp_path / f"bundle-{version}"
        out.mkdir()
        bundle = joblib.load(small_model_path)
        bundle["manifest"] = dict(bundle["manifest"], model_version=version)
        save_bundle(bundle, out / "model.joblib")
        return out
    return _make


@pytest.fixture()
def register(monkeypatch):
    """Register a bundle through the real ml/register_model.py CLI."""
    import ml.register_model as register_model

    def _register(path: Path, name: str, *, promote: bool) -> None:
        argv = ["register_model.py", "--bundle", str(path), "--model-name", name]
        if promote:
            argv.append("--promote")
        monkeypatch.setattr(sys, "argv", argv)
        assert register_model.main() == 0
    return _register


# --- champion resolution -------------------------------------------------

def test_resolves_champion_registered_by_promote(local_registry, bundle_dir,
                                                 register) -> None:
    """The pointer register_model.py --promote writes is the one we read."""
    from inference.registry import _client, resolve_champion_version

    register(bundle_dir("20260101.000000+aaaaaaa"), local_registry, promote=True)

    assert resolve_champion_version(_client(), local_registry) == "1"


def test_promoting_again_moves_the_pointer(local_registry, bundle_dir,
                                           register) -> None:
    from inference.registry import _client, resolve_champion_version

    register(bundle_dir("20260101.000000+aaaaaaa"), local_registry, promote=True)
    register(bundle_dir("20260202.000000+bbbbbbb"), local_registry, promote=True)

    assert resolve_champion_version(_client(), local_registry) == "2"


def test_challenger_alone_is_not_a_champion(local_registry, bundle_dir,
                                            register) -> None:
    from inference.registry import RegistryError, _client, resolve_champion_version

    register(bundle_dir("20260101.000000+aaaaaaa"), local_registry, promote=False)

    with pytest.raises(RegistryError, match="no champion"):
        resolve_champion_version(_client(), local_registry)


def test_falls_back_to_role_tag_when_pointer_absent(local_registry, bundle_dir,
                                                    register) -> None:
    """A registry without the model-level pointer still resolves, newest first."""
    from inference.registry import _client, resolve_champion_version

    register(bundle_dir("20260101.000000+aaaaaaa"), local_registry, promote=True)
    register(bundle_dir("20260202.000000+bbbbbbb"), local_registry, promote=True)
    # Simulate a registry where only per-version role tags exist.
    MlflowClient().delete_registered_model_tag(local_registry, "champion_version")

    assert resolve_champion_version(_client(), local_registry) == "2"


def test_unknown_model_raises(local_registry) -> None:
    from inference.registry import RegistryError, _client, resolve_champion_version

    with pytest.raises(RegistryError, match="not found in registry"):
        resolve_champion_version(_client(), "does-not-exist")


def test_missing_tracking_uri_raises(monkeypatch) -> None:
    from inference.registry import RegistryError, _client

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RegistryError, match="MLFLOW_TRACKING_URI"):
        _client()


def test_fetch_champion_downloads_the_promoted_bundle(local_registry, bundle_dir,
                                                      register) -> None:
    from inference.registry import fetch_champion

    register(bundle_dir("20260303.000000+ccccccc"), local_registry, promote=True)

    ref = fetch_champion(local_registry)
    assert ref.registry_version == "1"
    assert ref.path.exists() and ref.path.name == "model.joblib"
    assert joblib.load(ref.path)["manifest"]["model_version"] == \
        "20260303.000000+ccccccc"


# --- serving from the registry -------------------------------------------

@pytest.fixture()
def served_from_file(small_model_path, tmp_path, monkeypatch):
    """Server booted on the baked-in file bundle (the default source)."""
    live = tmp_path / "live-model.joblib"
    shutil.copy(small_model_path, live)
    monkeypatch.setenv("EDGESENSE_MODEL", str(live))
    monkeypatch.delenv("EDGESENSE_MODEL_SOURCE", raising=False)
    import inference.server as server
    importlib.reload(server)
    return TestClient(server.app)


def test_boots_from_registry_when_configured(local_registry, bundle_dir, register,
                                             small_model_path, monkeypatch) -> None:
    register(bundle_dir("20260404.000000+ddddddd"), local_registry, promote=True)

    monkeypatch.setenv("EDGESENSE_MODEL", str(small_model_path))
    monkeypatch.setenv("EDGESENSE_MODEL_SOURCE", "registry")
    monkeypatch.setenv("EDGESENSE_MODEL_NAME", local_registry)
    import inference.server as server
    importlib.reload(server)

    body = TestClient(server.app).get("/healthz").json()
    assert body["model_source"] == "registry"
    assert body["registry_version"] == "1"
    assert body["model_version"] == "20260404.000000+ddddddd"


def test_falls_back_to_file_when_registry_unreachable(small_model_path,
                                                      monkeypatch) -> None:
    """An edge node booting offline keeps serving its baked-in model — fast.

    MLflow's default retry policy (7 retries, backoff factor 2) would stall
    startup for ~4 minutes; `inference.registry` caps it so the fallback lands
    in seconds. The elapsed-time assertion is what keeps that capped.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # refused
    monkeypatch.setenv("EDGESENSE_MODEL", str(small_model_path))
    monkeypatch.setenv("EDGESENSE_MODEL_SOURCE", "registry")
    for var in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)  # exercise the shipped defaults

    import inference.server as server
    started = time.monotonic()
    importlib.reload(server)
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"registry fallback took {elapsed:.0f}s — retry cap regressed"

    client = TestClient(server.app)
    body = client.get("/healthz").json()
    assert body["model_source"] == "file"
    assert body["registry_version"] is None
    resp = client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                       "current": 12.0})
    assert resp.status_code == 200


def test_reload_from_registry_rolls_out_new_champion(served_from_file,
                                                     local_registry, bundle_dir,
                                                     register, monkeypatch) -> None:
    """The rollout path: promote in the registry, then reload the sidecar."""
    client = served_from_file
    before = client.get("/healthz").json()
    assert before["model_source"] == "file"

    register(bundle_dir("20260505.000000+eeeeeee"), local_registry, promote=True)
    import inference.server as server
    monkeypatch.setattr(server, "MODEL_NAME", local_registry)

    resp = client.post("/reload?source=registry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["old_version"] == before["model_version"]
    assert body["new_version"] == "20260505.000000+eeeeeee"
    assert body["source"] == "registry" and body["registry_version"] == "1"

    health = client.get("/healthz").json()
    assert health["model_source"] == "registry"
    assert health["model_version"] == "20260505.000000+eeeeeee"
    assert client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                       "current": 12.0}).status_code == 200


def test_failed_registry_reload_keeps_champion_serving(served_from_file,
                                                       local_registry,
                                                       monkeypatch) -> None:
    """Nothing promoted yet: the reload is refused and the old model serves on."""
    client = served_from_file
    before = client.get("/healthz").json()

    import inference.server as server
    monkeypatch.setattr(server, "MODEL_NAME", local_registry)

    resp = client.post("/reload?source=registry")
    assert resp.status_code == 400
    assert before["model_version"] in resp.json()["detail"]

    after = client.get("/healthz").json()
    assert after["model_version"] == before["model_version"]
    assert after["model_source"] == "file"
    assert client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                       "current": 12.0}).status_code == 200


def test_unknown_source_rejected(served_from_file) -> None:
    resp = served_from_file.post("/reload?source=s3")
    assert resp.status_code == 400
    assert "expected file|registry" in resp.json()["detail"]
