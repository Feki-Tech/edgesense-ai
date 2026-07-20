"""EdgeSense live dashboard (Streamlit).

Subscribes to sensor readings (local broker) and anomaly events (uplink
broker — same as local unless EDGESENSE_UPLINK_HOST/PORT differ) over MQTT
and renders live signal plots with anomaly markers plus an event feed.

Topic layout follows EDGESENSE_ORG/EDGESENSE_SITE: unset → legacy flat
topics; set → tenant-namespaced `es/<org>/<site>/…` (PLATFORM.md §4.4).
EDGESENSE_UPLINK_USERNAME/_PASSWORD authenticate against a secured uplink
broker.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import paho.mqtt.client as mqtt

BROKER = os.environ.get("EDGESENSE_BROKER_HOST", "localhost")
PORT = int(os.environ.get("EDGESENSE_BROKER_PORT", "11883"))
UPLINK = os.environ.get("EDGESENSE_UPLINK_HOST", BROKER)
UPLINK_PORT = int(os.environ.get("EDGESENSE_UPLINK_PORT", str(PORT)))
UPLINK_AUTH = (os.environ.get("EDGESENSE_UPLINK_USERNAME"),
               os.environ.get("EDGESENSE_UPLINK_PASSWORD", ""))
ORG = os.environ.get("EDGESENSE_ORG")
SITE = os.environ.get("EDGESENSE_SITE")
if ORG or SITE:
    _PREFIX = f"es/{ORG or 'default'}/{SITE or 'default'}"
    SENSOR_TOPICS = [f"{_PREFIX}/+/sensors/#"]
    EVENT_TOPICS = [f"{_PREFIX}/+/events"]
else:
    SENSOR_TOPICS = ["edgesense/sensors/#"]
    EVENT_TOPICS = ["edgesense/events/#"]
MAX_POINTS = 600
SIGNALS = ["vibration", "temperature", "current"]


def is_reading_topic(topic: str) -> bool:
    """True for sensor-reading topics in either layout (else: event)."""
    parts = topic.split("/")
    if parts[0] == "es":
        return len(parts) >= 5 and parts[4] == "sensors"
    return len(parts) >= 2 and parts[1] == "sensors"


class Collector:
    """Background MQTT subscriber accumulating readings and events."""

    def __init__(self, endpoints: list[tuple[str, int, list[str], tuple | None]]) -> None:
        self.lock = threading.Lock()
        self.readings: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_POINTS))
        self.events: deque = deque(maxlen=200)
        self.clients: list[mqtt.Client] = []

        for i, (host, port, topics, auth) in enumerate(endpoints):
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"edgesense-dashboard-{i}-{int(time.time())}")
            if auth and auth[0]:
                client.username_pw_set(auth[0], auth[1])
            client.on_connect = self._make_on_connect(topics)
            client.on_message = self._on_message
            client.connect(host, port)
            client.loop_start()
            self.clients.append(client)

    @staticmethod
    def _make_on_connect(topics: list[str]):
        def on_connect(client, *_args) -> None:
            client.subscribe([(t, 0) for t in topics])
        return on_connect

    def _on_message(self, _client, _userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        with self.lock:
            if is_reading_topic(msg.topic):
                self.readings[payload.get("machine_id", "?")].append(payload)
            else:
                self.events.appendleft(payload)

    def snapshot(self) -> tuple[dict[str, pd.DataFrame], list[dict]]:
        with self.lock:
            frames = {m: pd.DataFrame(list(d)) for m, d in self.readings.items() if d}
            events = list(self.events)
        return frames, events


@st.cache_resource
def get_collector() -> Collector:
    if (BROKER, PORT) == (UPLINK, UPLINK_PORT):
        endpoints = [(BROKER, PORT, SENSOR_TOPICS + EVENT_TOPICS, UPLINK_AUTH)]
    else:
        endpoints = [(BROKER, PORT, SENSOR_TOPICS, None),
                     (UPLINK, UPLINK_PORT, EVENT_TOPICS, UPLINK_AUTH)]
    return Collector(endpoints)


st.set_page_config(page_title="EdgeSense AI", layout="wide")
st_autorefresh(interval=2000, key="refresh")
st.title("EdgeSense AI — live machine monitoring")

collector = get_collector()
frames, events = collector.snapshot()

if not frames:
    st.info("Waiting for sensor data… start the broker, simulator, inference and agent.")
    st.stop()

machines = sorted(frames)
selected = st.sidebar.multiselect("Machines", machines, default=machines)
st.sidebar.metric("Machines online", len(machines))
st.sidebar.metric("Anomaly events", len(events))

event_ts = defaultdict(list)
for ev in events:
    event_ts[ev.get("machine_id", "?")].append(ev.get("ts"))

cols = st.columns(len(SIGNALS))
for col, signal_name in zip(cols, SIGNALS):
    fig = go.Figure()
    for m in selected:
        df = frames[m]
        fig.add_trace(go.Scatter(x=pd.to_datetime(df["ts"], unit="s"),
                                 y=df[signal_name], mode="lines", name=m))
        hits = df[df["ts"].isin(event_ts.get(m, []))]
        if not hits.empty:
            fig.add_trace(go.Scatter(x=pd.to_datetime(hits["ts"], unit="s"),
                                     y=hits[signal_name], mode="markers",
                                     marker=dict(color="red", size=9, symbol="x"),
                                     name=f"{m} anomaly", showlegend=False))
    fig.update_layout(title=signal_name, height=320,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h"))
    col.plotly_chart(fig, use_container_width=True)

st.subheader("Anomaly event feed")
if events:
    ev_df = pd.DataFrame(events)
    ev_df["time"] = pd.to_datetime(ev_df["ts"], unit="s").dt.strftime("%H:%M:%S")
    reading_df = pd.json_normalize(ev_df["reading"])
    cols_to_show = [c for c in ("time", "machine_id", "score", "reason") if c in ev_df]
    show = pd.concat(
        [ev_df[cols_to_show],
         reading_df[["vibration", "temperature", "current"]]], axis=1)
    st.dataframe(show, use_container_width=True, height=280)
else:
    st.caption("No anomalies detected yet.")
