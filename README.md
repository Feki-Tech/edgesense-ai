# EdgeSense AI

On-device anomaly detection for industrial machines:
sensors at the edge, ML inference on the node, and only *events* go upstream.

```
┌────────────┐  MQTT   ┌─────────────┐  HTTP   ┌──────────────┐
│ simulator  │ ──────► │ edge-agent  │ ──────► │ inference    │
│ (sensors)  │ sensors │ (Go)        │ /score  │ (FastAPI +   │
└────────────┘         │             │ ◄────── │  IsolationF.)│
                       └─────┬───────┘         └──────────────┘
                             │ MQTT: edgesense/events/#  (anomalies only)
                             ▼
                       ┌────────────┐
                       │ dashboard  │  (Streamlit, live plots + alerts)
                       └────────────┘
```

## Components

| Path         | Role                                                              |
|--------------|-------------------------------------------------------------------|
| `simulator/` | Simulates machines publishing vibration / temperature / current over MQTT, with injected fault episodes (bearing fault, overheat, overload). |
| `ml/`        | Trains an IsolationForest on normal operating data → `ml/model/model.joblib`. |
| `inference/` | FastAPI sidecar serving the model (`POST /score`).                 |
| `edge-agent/`| Go agent: subscribes to sensor topics, scores each reading, publishes anomaly events. |
| `dashboard/` | Streamlit live dashboard: signals, anomaly scores, event feed.     |
| `deploy/`    | Mosquitto broker config (docker compose).                          |

## Quickstart

```bash
make setup        # venv + python deps + go deps
make broker       # start mosquitto (docker)
make train        # train + validate the anomaly model

# in separate terminals:
make inference    # :8800
make agent
make simulate
make dashboard    # :8501

make smoke        # end-to-end check (broker + inference + event round-trip)
```

## MQTT topics

- `edgesense/sensors/<machine_id>` — raw readings (JSON), ~2 Hz per machine
- `edgesense/events/<machine_id>` — anomaly events only (JSON, with score)

## Ideas / roadmap

- Package agent + inference as snaps (Ubuntu Core), model updates as snap refreshes
- Replace IsolationForest with an autoencoder, export to ONNX for the Go agent (drop the sidecar)
- CoAP uplink for constrained/LTE links, store-and-forward buffering
- Fleet view: many virtual devices via docker compose scale
