"""Inference API tests via FastAPI TestClient."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(small_model_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("EDGESENSE_MODEL", str(small_model_path))
    import inference.server as server
    importlib.reload(server)
    return TestClient(server.app)


def test_healthz(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["features"] == ["vibration", "temperature", "current"]


def test_score_healthy_reading(client: TestClient) -> None:
    resp = client.post("/score", json={"vibration": 0.8, "temperature": 45.0, "current": 12.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_anomaly"] is False
    assert body["score"] > 0


def test_score_faulty_reading(client: TestClient) -> None:
    resp = client.post("/score", json={"vibration": 4.2, "temperature": 46.0, "current": 14.5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_anomaly"] is True
    assert body["reason"] in ("model", "limit", "model+limit")


def test_single_feature_outlier_flagged_by_guard(client: TestClient) -> None:
    resp = client.post("/score", json={"vibration": 0.8, "temperature": 78.0, "current": 12.0})
    body = resp.json()
    assert body["is_anomaly"] is True


def test_score_missing_field_rejected(client: TestClient) -> None:
    resp = client.post("/score", json={"vibration": 0.8, "temperature": 45.0})
    assert resp.status_code == 422
