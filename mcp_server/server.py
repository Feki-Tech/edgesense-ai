"""EdgeSense MCP server.

Exposes the EdgeSense stack to MCP clients (Claude Desktop, IDEs, agents)
via the Model Context Protocol: score readings against the inference
sidecar, check its health, inject demo faults over MQTT, tail recent
anomaly events from the uplink broker and query fleet metrics from
Prometheus.

Runs over stdio by default (the usual transport for local MCP clients);
the docker-compose service uses streamable-http on port 8900 instead:

    python mcp_server/server.py                  # stdio
    EDGESENSE_MCP_TRANSPORT=streamable-http python mcp_server/server.py
                                                 # -> http://localhost:8900/mcp

Configuration (same defaults as the host-side scripts / docker-compose):

    EDGESENSE_BROKER_HOST / EDGESENSE_BROKER_PORT     local broker (fault injection)
    EDGESENSE_UPLINK_HOST / EDGESENSE_UPLINK_PORT     uplink broker (anomaly events)
    EDGESENSE_INFERENCE_URL                            inference base URL
    EDGESENSE_PROMETHEUS_URL                           Prometheus base URL
    EDGESENSE_MCP_TRANSPORT / _HOST / _PORT            server transport & bind
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import paho.mqtt.client as mqtt
import requests
from mcp.server.fastmcp import FastMCP

BROKER = os.environ.get("EDGESENSE_BROKER_HOST", "localhost")
PORT = int(os.environ.get("EDGESENSE_BROKER_PORT", "11883"))
UPLINK = os.environ.get("EDGESENSE_UPLINK_HOST", BROKER)
UPLINK_PORT = int(os.environ.get("EDGESENSE_UPLINK_PORT", "12883"))
PROMETHEUS_URL = os.environ.get("EDGESENSE_PROMETHEUS_URL", "http://localhost:9090").rstrip("/")

INFERENCE_URL = os.environ.get("EDGESENSE_INFERENCE_URL", "http://localhost:8800").rstrip("/")
if INFERENCE_URL.endswith("/score"):  # tolerate the agent-style URL (…:8800/score)
    INFERENCE_URL = INFERENCE_URL[: -len("/score")]

TRANSPORT = os.environ.get("EDGESENSE_MCP_TRANSPORT", "stdio")
CONTROL_TOPIC = "edgesense/control/fault"
EVENTS_TOPIC = "edgesense/events/#"
FAULT_TYPES = ("bearing_fault", "overheat", "overload", "clear")
MAX_EVENTS = 500
HTTP_TIMEOUT = 5.0

mcp = FastMCP(
    "EdgeSense AI",
    instructions="Tools for the EdgeSense edge anomaly-detection stack: "
                 "score sensor readings, inject demo faults, tail anomaly "
                 "events and query fleet metrics.",
    host=os.environ.get("EDGESENSE_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("EDGESENSE_MCP_PORT", "8900")),
)


class EventCollector:
    """Background MQTT subscriber accumulating anomaly events (newest first)."""

    def __init__(self, endpoints: list[tuple[str, int]]) -> None:
        self._endpoints = endpoints
        self.lock = threading.Lock()
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.clients: list[mqtt.Client] = []

    def start(self) -> None:
        for i, (host, port) in enumerate(self._endpoints):
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"edgesense-mcp-{i}-{int(time.time())}")
            client.on_connect = lambda c, *_: c.subscribe(EVENTS_TOPIC)
            client.on_message = self._on_message
            client.connect(host, port)
            client.loop_start()
            self.clients.append(client)

    def _on_message(self, _client, _userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.events.appendleft(payload)

    def recent(self, limit: int, machine_id: str | None = None) -> list[dict]:
        with self.lock:
            events = list(self.events)
        if machine_id is not None:
            events = [e for e in events if e.get("machine_id") == machine_id]
        return events[:limit]


_collector: EventCollector | None = None
_collector_lock = threading.Lock()


def _get_collector() -> EventCollector:
    """Start the event subscriber on first use and reuse it afterwards."""
    global _collector
    with _collector_lock:
        if _collector is None:
            collector = EventCollector([(UPLINK, UPLINK_PORT)])
            collector.start()
            _collector = collector
    return _collector


def _prom_query(expr: str) -> float | None:
    """Run an instant PromQL query; return the scalar result or None if empty."""
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                        params={"query": expr}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    results = body.get("data", {}).get("result", [])
    if body.get("status") != "success" or not results:
        return None
    return float(results[0]["value"][1])


@mcp.tool()
def score_reading(vibration: float, temperature: float, current: float) -> dict:
    """Score one sensor reading against the EdgeSense anomaly model.

    Returns {"score", "is_anomaly", "reason"} from the inference sidecar —
    score is the autoencoder's reconstruction error (higher = more
    anomalous); reason is "model", "limit" or "model+limit".
    """
    resp = requests.post(
        f"{INFERENCE_URL}/score",
        json={"vibration": vibration, "temperature": temperature, "current": current},
        timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def get_inference_health() -> dict:
    """Check the inference sidecar: model path, kind and feature list."""
    resp = requests.get(f"{INFERENCE_URL}/healthz", timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def inject_fault(machine_id: str, fault: str, ticks: int = 24) -> dict:
    """Inject a fault episode into a simulated machine (or clear one).

    Publishes to the edgesense/control/fault topic on the local broker,
    exactly like the scripted demos. fault is one of bearing_fault,
    overheat, overload or clear; ticks is the episode length in readings
    (~2 per second).
    """
    if fault not in FAULT_TYPES:
        raise ValueError(f"unknown fault {fault!r}; expected one of {', '.join(FAULT_TYPES)}")
    payload = {"machine_id": machine_id, "fault": fault, "ticks": ticks}
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"edgesense-mcp-fault-{int(time.time())}")
    client.connect(BROKER, PORT)
    client.loop_start()
    try:
        info = client.publish(CONTROL_TOPIC, json.dumps(payload), qos=1)
        info.wait_for_publish(timeout=HTTP_TIMEOUT)
    finally:
        client.loop_stop()
        client.disconnect()
    return {"published": True, "topic": CONTROL_TOPIC, **payload}


@mcp.tool()
def list_recent_events(limit: int = 20, machine_id: str | None = None) -> list[dict]:
    """List recent anomaly events from the uplink broker, newest first.

    Each event carries machine_id, ts, score, reason and the offending
    reading. Collection starts with the first call of this tool, so it
    only sees events raised after that point (up to 500 are retained).
    """
    return _get_collector().recent(max(1, limit), machine_id)


@mcp.tool()
def get_fleet_metrics() -> dict:
    """Summarise fleet health from Prometheus (agent metrics).

    Returns readings/anomalies per second (1m rates), the anomaly rate,
    the store-and-forward buffer depth, uplink status (min over agents:
    1 = all connected, 0 = at least one down) and total events published.
    Values are null when Prometheus has no data yet.
    """
    readings_rate = _prom_query("sum(rate(edgesense_readings_scored_total[1m]))")
    anomalies_rate = _prom_query("sum(rate(edgesense_anomalies_total[1m]))")
    anomaly_rate = (anomalies_rate / readings_rate
                    if readings_rate and anomalies_rate is not None else None)
    return {
        "readings_per_second": readings_rate,
        "anomalies_per_second": anomalies_rate,
        "anomaly_rate": anomaly_rate,
        "buffer_depth": _prom_query("sum(edgesense_buffer_depth)"),
        "uplink_connected": _prom_query("min(edgesense_uplink_connected)"),
        "events_published_total": _prom_query("sum(edgesense_events_published_total)"),
    }


def main() -> None:
    """Run the MCP server on the configured transport."""
    mcp.run(transport=TRANSPORT)


if __name__ == "__main__":
    main()
