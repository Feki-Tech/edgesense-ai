"""Shadow scoring: agreement tracking, failure isolation, lifecycle."""

from __future__ import annotations

import importlib
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from ml.manifest import save_bundle

NOMINAL = {"vibration": 0.8, "temperature": 45.0, "current": 12.0}


@pytest.fixture()
def served(small_model_path, tmp_path, monkeypatch):
    """A client serving the small bundle, with a swappable candidate path."""
    live_path = tmp_path / "model.joblib"
    candidate_path = tmp_path / "candidate" / "model.joblib"
    candidate_path.parent.mkdir()
    shutil.copy(small_model_path, live_path)
    monkeypatch.setenv("EDGESENSE_MODEL", str(live_path))
    monkeypatch.setenv("EDGESENSE_SHADOW_MODEL", str(candidate_path))
    import inference.server as server
    importlib.reload(server)
    return TestClient(server.app), live_path, candidate_path


def _candidate(small_model_path, version: str = "20990101.000000+cand0000",
               **overrides) -> dict:
    bundle = joblib.load(small_model_path)
    bundle["manifest"] = dict(bundle["manifest"], model_version=version)
    bundle.update(overrides)
    return bundle


def test_no_shadow_by_default(served) -> None:
    client, _, _ = served
    assert client.get("/shadow").json() == {"active": False}
    assert client.post("/shadow/unload").status_code == 404
    # scoring works without a shadow
    assert client.post("/score", json=NOMINAL).status_code == 200


def test_shadow_load_missing_candidate_is_rejected(served) -> None:
    client, _, _ = served
    assert client.post("/shadow/load").status_code == 400
    assert client.get("/shadow").json() == {"active": False}


def test_shadow_agreement_accumulates(served, small_model_path) -> None:
    client, _, candidate_path = served
    save_bundle(_candidate(small_model_path), candidate_path)

    body = client.post("/shadow/load").json()
    assert body["status"] == "shadowing"
    assert body["shadow_version"] == "20990101.000000+cand0000"

    for _ in range(3):
        resp = client.post("/score", json=NOMINAL)
        assert resp.status_code == 200 and resp.json()["is_anomaly"] is False

    report = client.get("/shadow").json()["report"]
    assert report["n"] == 3 and report["agree"] == 3
    assert report["agreement_rate"] == 1.0
    assert report["score_mae"] == pytest.approx(0.0)  # identical model


def test_shadow_disagreement_is_counted(served, small_model_path) -> None:
    client, _, candidate_path = served
    # A hair-trigger threshold makes the shadow flag readings the champion passes.
    save_bundle(_candidate(small_model_path, threshold=1e-12), candidate_path)
    client.post("/shadow/load")

    resp = client.post("/score", json=NOMINAL)
    # the champion alone decides the response
    assert resp.json()["is_anomaly"] is False

    report = client.get("/shadow").json()["report"]
    assert report["shadow_only"] == 1 and report["agree"] == 0
    assert report["agreement_rate"] == 0.0


def test_shadow_errors_never_break_scoring(served, small_model_path) -> None:
    client, _, candidate_path = served
    # Valid at load time, but its feature list doesn't match live readings.
    save_bundle(_candidate(small_model_path,
                           features=["vibration", "temperature", "bogus"]),
                candidate_path)
    client.post("/shadow/load")

    resp = client.post("/score", json=NOMINAL)
    assert resp.status_code == 200

    report = client.get("/shadow").json()["report"]
    assert report["errors"] == 1 and report["n"] == 0


def test_shadow_unload_returns_final_report(served, small_model_path) -> None:
    client, _, candidate_path = served
    save_bundle(_candidate(small_model_path), candidate_path)
    client.post("/shadow/load")
    client.post("/score", json=NOMINAL)

    body = client.post("/shadow/unload").json()
    assert body["status"] == "unloaded" and body["report"]["n"] == 1
    assert client.get("/shadow").json() == {"active": False}


def test_champion_reload_resets_shadow_evidence(served, small_model_path) -> None:
    client, live_path, candidate_path = served
    save_bundle(_candidate(small_model_path), candidate_path)
    client.post("/shadow/load")
    client.post("/score", json=NOMINAL)
    assert client.get("/shadow").json()["report"]["n"] == 1

    save_bundle(_candidate(small_model_path, "20990102.000000+champ000"), live_path)
    assert client.post("/reload").status_code == 200

    report = client.get("/shadow").json()["report"]
    assert report["n"] == 0  # per-champion evidence starts over
    assert report["champion_version"] == "20990102.000000+champ000"
    assert report["shadow_version"] == "20990101.000000+cand0000"
