"""Hot model reload: atomic swap, validation, and failure isolation."""

from __future__ import annotations

import importlib
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from ml.manifest import save_bundle


@pytest.fixture()
def served(small_model_path, tmp_path, monkeypatch):
    """A client serving a copy of the small bundle from a swappable path."""
    live_path = tmp_path / "model.joblib"
    shutil.copy(small_model_path, live_path)
    monkeypatch.setenv("EDGESENSE_MODEL", str(live_path))
    import inference.server as server
    importlib.reload(server)
    return TestClient(server.app), live_path


def _versioned_copy(small_model_path, version: str) -> dict:
    bundle = joblib.load(small_model_path)
    bundle["manifest"] = dict(bundle["manifest"], model_version=version)
    return bundle


def test_healthz_reports_model_version(served, small_model_path) -> None:
    client, _ = served
    body = client.get("/healthz").json()
    expected = joblib.load(small_model_path)["manifest"]
    assert body["model_version"] == expected["model_version"]
    assert body["created_at"] == expected["created_at"]


def test_reload_swaps_versions(served, small_model_path) -> None:
    client, live_path = served
    old_version = client.get("/healthz").json()["model_version"]

    save_bundle(_versioned_copy(small_model_path, "20990101.000000+abcdef0"), live_path)

    resp = client.post("/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert body["old_version"] == old_version
    assert body["new_version"] == "20990101.000000+abcdef0"
    assert client.get("/healthz").json()["model_version"] == "20990101.000000+abcdef0"

    # the swapped-in model serves scores
    resp = client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                       "current": 12.0})
    assert resp.status_code == 200 and resp.json()["is_anomaly"] is False


def test_reload_rejects_garbage_file_and_keeps_serving(served) -> None:
    client, live_path = served
    old_version = client.get("/healthz").json()["model_version"]

    live_path.write_bytes(b"this is not a joblib bundle")
    resp = client.post("/reload")
    assert resp.status_code == 400
    assert old_version in resp.json()["detail"]

    # old model still serves, /healthz unchanged
    assert client.get("/healthz").json()["model_version"] == old_version
    resp = client.post("/score", json={"vibration": 4.2, "temperature": 46.0,
                                       "current": 14.5})
    assert resp.status_code == 200 and resp.json()["is_anomaly"] is True


def test_reload_rejects_invalid_bundle_shape(served, small_model_path) -> None:
    client, live_path = served
    bundle = joblib.load(small_model_path)
    bundle["weights"] = bundle["weights"][:-1]  # broken layer chain
    joblib.dump(bundle, live_path)

    assert client.post("/reload").status_code == 400


def test_reload_missing_file_is_rejected(served) -> None:
    client, live_path = served
    live_path.unlink()
    assert client.post("/reload").status_code == 400
    assert client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                       "current": 12.0}).status_code == 200


def test_reload_accepts_legacy_bundle_without_manifest(served, legacy_model_path) -> None:
    client, live_path = served
    shutil.copy(legacy_model_path, live_path)
    body = client.post("/reload").json()
    assert body["new_version"] == "unknown"
    assert client.get("/healthz").json()["model_version"] == "unknown"


def test_reload_resets_drift_window(served, small_model_path) -> None:
    import inference.server as server
    client, live_path = served
    for _ in range(60):
        client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                    "current": 12.0})
    assert server._drift.size > 0

    save_bundle(_versioned_copy(small_model_path, "20990101.000000+abcdef0"), live_path)
    assert client.post("/reload").status_code == 200
    assert server._drift.size == 0
