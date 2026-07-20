"""MCP server tool tests (HTTP and MQTT mocked — no stack required)."""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest


@pytest.fixture()
def server(monkeypatch):
    """Fresh mcp_server.server module with default env."""
    for var in ("EDGESENSE_BROKER_HOST", "EDGESENSE_BROKER_PORT",
                "EDGESENSE_UPLINK_HOST", "EDGESENSE_UPLINK_PORT",
                "EDGESENSE_INFERENCE_URL", "EDGESENSE_PROMETHEUS_URL"):
        monkeypatch.delenv(var, raising=False)
    import mcp_server.server as srv
    importlib.reload(srv)
    return srv


class FakeResponse:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = body
        self.status_code = status

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeMsg:
    def __init__(self, topic: str, payload) -> None:
        self.topic = topic
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()


class FakePublishInfo:
    def wait_for_publish(self, timeout=None) -> None:
        pass


class FakeMqttClient:
    """Stands in for paho.mqtt.client.Client; records connects and publishes."""

    instances: list["FakeMqttClient"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.connected_to: tuple[str, int] | None = None
        self.published: list[tuple[str, str, int]] = []
        self.on_connect = None
        self.on_message = None
        FakeMqttClient.instances.append(self)

    def connect(self, host: str, port: int) -> None:
        self.connected_to = (host, port)

    def publish(self, topic: str, payload: str, qos: int = 0) -> FakePublishInfo:
        self.published.append((topic, payload, qos))
        return FakePublishInfo()

    def subscribe(self, *_args) -> None:
        pass

    def loop_start(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


def test_tools_registered(server) -> None:
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"score_reading", "get_inference_health", "inject_fault",
                     "list_recent_events", "get_fleet_metrics"}


def test_score_reading_proxies_inference(server, monkeypatch) -> None:
    calls = {}

    def fake_post(url, json=None, timeout=None):
        calls["url"], calls["json"], calls["timeout"] = url, json, timeout
        return FakeResponse({"score": 38.2, "is_anomaly": True, "reason": "model"})

    monkeypatch.setattr(server.requests, "post", fake_post)
    result = server.score_reading(vibration=4.2, temperature=46.0, current=14.5)
    assert calls["url"] == "http://localhost:8800/score"
    assert calls["json"] == {"vibration": 4.2, "temperature": 46.0, "current": 14.5}
    assert calls["timeout"] == server.HTTP_TIMEOUT
    assert result == {"score": 38.2, "is_anomaly": True, "reason": "model"}


def test_inference_url_score_suffix_trimmed(monkeypatch) -> None:
    # the agent-style URL from docker-compose must work unchanged
    monkeypatch.setenv("EDGESENSE_INFERENCE_URL", "http://inference:8800/score")
    import mcp_server.server as srv
    importlib.reload(srv)
    assert srv.INFERENCE_URL == "http://inference:8800"


def test_get_inference_health(server, monkeypatch) -> None:
    body = {"status": "ok", "model_kind": "autoencoder",
            "features": ["vibration", "temperature", "current"]}
    monkeypatch.setattr(server.requests, "get",
                        lambda url, timeout=None: FakeResponse(body)
                        if url == "http://localhost:8800/healthz" else FakeResponse({}, 404))
    assert server.get_inference_health() == body


def test_inject_fault_publishes_demo_payload(server, monkeypatch) -> None:
    FakeMqttClient.instances = []
    monkeypatch.setattr(server.mqtt, "Client", FakeMqttClient)
    result = server.inject_fault(machine_id="machine-02", fault="overheat", ticks=30)

    client = FakeMqttClient.instances[0]
    assert client.connected_to == ("localhost", 11883)
    topic, payload, qos = client.published[0]
    assert topic == "edgesense/control/fault"
    assert qos == 1
    # exact payload format used by scripts/demo.py and the simulator
    assert json.loads(payload) == {"machine_id": "machine-02", "fault": "overheat", "ticks": 30}
    assert result["published"] is True


def test_inject_fault_rejects_unknown_fault(server, monkeypatch) -> None:
    monkeypatch.setattr(server.mqtt, "Client", FakeMqttClient)
    with pytest.raises(ValueError, match="unknown fault"):
        server.inject_fault(machine_id="machine-01", fault="meltdown")


def test_event_collector_orders_filters_and_limits(server) -> None:
    collector = server.EventCollector([])  # never started -> no sockets
    for i in range(5):
        machine = f"machine-{i % 2:02d}"
        collector._on_message(None, None, FakeMsg(
            f"edgesense/events/{machine}",
            {"machine_id": machine, "ts": float(i), "score": 30.0 + i, "reason": "model"}))
    collector._on_message(None, None, FakeMsg("edgesense/events/x", b"not json"))

    recent = collector.recent(limit=3)
    assert [e["ts"] for e in recent] == [4.0, 3.0, 2.0]  # newest first
    only_00 = collector.recent(limit=10, machine_id="machine-00")
    assert {e["machine_id"] for e in only_00} == {"machine-00"}
    assert len(only_00) == 3


def test_list_recent_events_uses_shared_collector(server) -> None:
    collector = server.EventCollector([])
    collector._on_message(None, None, FakeMsg(
        "edgesense/events/machine-07",
        {"machine_id": "machine-07", "ts": 1.0, "score": 55.1, "reason": "limit"}))
    server._collector = collector

    events = server.list_recent_events(limit=5)
    assert events == [{"machine_id": "machine-07", "ts": 1.0, "score": 55.1, "reason": "limit"}]
    assert server.list_recent_events(limit=5, machine_id="machine-99") == []


def test_get_fleet_metrics_queries_prometheus(server, monkeypatch) -> None:
    values = {
        "sum(rate(edgesense_readings_scored_total[1m]))": "6.0",
        "sum(rate(edgesense_anomalies_total[1m]))": "0.3",
        "sum(edgesense_buffer_depth)": "12",
        "min(edgesense_uplink_connected)": "1",
        "sum(edgesense_events_published_total)": "240",
    }

    def fake_get(url, params=None, timeout=None):
        assert url == "http://localhost:9090/api/v1/query"
        value = values[params["query"]]
        return FakeResponse({"status": "success",
                             "data": {"result": [{"value": [1e9, value]}]}})

    monkeypatch.setattr(server.requests, "get", fake_get)
    metrics = server.get_fleet_metrics()
    assert metrics["readings_per_second"] == 6.0
    assert metrics["anomalies_per_second"] == 0.3
    assert metrics["anomaly_rate"] == pytest.approx(0.05)
    assert metrics["buffer_depth"] == 12.0
    assert metrics["uplink_connected"] == 1.0
    assert metrics["events_published_total"] == 240.0


def test_get_fleet_metrics_handles_empty_prometheus(server, monkeypatch) -> None:
    monkeypatch.setattr(
        server.requests, "get",
        lambda url, params=None, timeout=None: FakeResponse(
            {"status": "success", "data": {"result": []}}))
    metrics = server.get_fleet_metrics()
    assert all(v is None for v in metrics.values())
