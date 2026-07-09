"""EdgeSense live dashboard (Streamlit).

Subscribes to sensor readings and anomaly events over MQTT and renders
live signal plots with anomaly markers plus an event feed.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
MAX_POINTS = 600
SIGNALS = ["vibration", "temperature", "current"]


class Collector:
    """Background MQTT subscriber accumulating readings and events."""

    def __init__(self, broker: str, port: int) -> None:
        self.lock = threading.Lock()
        self.readings: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_POINTS))
        self.events: deque = deque(maxlen=200)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"edgesense-dashboard-{int(time.time())}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker, port)
        self.client.loop_start()

    def _on_connect(self, client, *_args) -> None:
        client.subscribe([("edgesense/sensors/#", 0), ("edgesense/events/#", 0)])

    def _on_message(self, _client, _userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        with self.lock:
            if msg.topic.startswith("edgesense/sensors/"):
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
    return Collector(BROKER, PORT)


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
    show = pd.concat(
        [ev_df[["time", "machine_id", "score"]],
         reading_df[["vibration", "temperature", "current"]]], axis=1)
    st.dataframe(show, use_container_width=True, height=280)
else:
    st.caption("No anomalies detected yet.")
