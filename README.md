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
| `ml/`        | Trains an IsolationForest on normal operating data → `ml/model/model.joblib`; hybrid scoring helpers (`scoring.py`); ONNX export (`export_onnx.py`). |
| `inference/` | FastAPI sidecar serving the model (`POST /score`).                 |
| `edge-agent/`| Go agent: subscribes to sensor topics, scores each reading, publishes anomaly events with store-and-forward buffering. |
| `dashboard/` | Streamlit live dashboard: signals, anomaly scores, event feed.     |
| `deploy/`    | Mosquitto broker config (docker compose).                          |
| `snap/`      | Snapcraft packaging for the edge agent (Ubuntu Core ready).        |

Every service ships a Dockerfile; `docker-compose.yml` wires them together
on an internal network (broker hostname `mosquitto`).

## Quickstart (Docker, recommended)

```bash
make stack        # build + start broker, inference, agent, simulator, dashboard
make stack-logs   # follow logs
make smoke        # end-to-end check from the host (needs `make setup` once)
make stack-down   # stop everything
```

- Dashboard: http://localhost:8501
- Inference API: http://localhost:8800/healthz
- Broker (host access): localhost:11883

The inference image bakes a freshly trained model at build time; the agent is
a distroless static Go binary with its event buffer on a named volume
(`agent-data`), so buffered events survive container restarts.

## Quickstart (local processes)

```bash
make setup        # venv + python deps (incl. dev) + go deps
make broker       # start mosquitto (docker)
make train        # train + validate the anomaly model

# in separate terminals:
make inference    # :8800
make agent
make simulate
make dashboard    # :8501

make smoke        # end-to-end check (broker + inference + event round-trip)
```

## Anomaly detection

Scoring is hybrid (`ml/scoring.py`): a reading is anomalous if the
IsolationForest flags it **or** any feature deviates more than `z_guard`
(default 6σ) from the training distribution. The guard catches single-feature
outliers (e.g. pure overheat) that isolation forests systematically miss.
Responses carry a `reason` field: `model`, `limit`, or `model+limit`.

## Store-and-forward

The agent publishes events with QoS 1. If the broker is unreachable, events
are appended to a disk-backed FIFO (`EDGESENSE_BUFFER`, JSON Lines, capped at
10k entries, oldest dropped first) and flushed on reconnect plus every 30 s.
Events survive agent restarts.

## ONNX export

```bash
make export-onnx   # -> ml/model/model.onnx + model.onnx.json (scaler + guard params)
```

`tests/test_onnx.py` asserts parity between onnxruntime and sklearn
(score MAE < 1e-3, label agreement > 99%). The exported model plus the
sidecar-free metadata file are the path to running inference directly in the
Go agent (roadmap).

## Snap packaging

`snap/snapcraft.yaml` packages the agent as a strictly-confined daemon
(core24, Go plugin, auto-restart, buffer in `$SNAP_COMMON`). Build on a
machine with snapcraft:

```bash
sudo snap install snapcraft --classic
make snap          # or: snapcraft pack
sudo snap install ./edgesense-agent_0.1.0_amd64.snap --dangerous
```

## Testing & CI

```bash
make test         # pytest (model quality, API, simulator, ONNX parity) + go test (agent + buffer)
```

GitHub Actions (`.github/workflows/ci.yml`) runs both suites on every push
and pull request: a Python 3.12 job (`pytest`) and a Go 1.22 job
(`go vet`, `go build`, `go test`).

## MQTT topics

The broker listens on host port **11883** (1883 sits in a Windows/Hyper-V
reserved port range when running under WSL2 + Docker Desktop).

- `edgesense/sensors/<machine_id>` — raw readings (JSON), ~2 Hz per machine
- `edgesense/events/<machine_id>` — anomaly events only (JSON, with score + reason)

## Ideas / roadmap

- Run ONNX inference inside the Go agent (onnxruntime bindings), drop the sidecar
- Replace IsolationForest with an autoencoder for richer fault signatures
- CoAP uplink for constrained/LTE links
- Fleet view: many virtual devices via docker compose scale
- Inference service as a second snap; model updates as snap refreshes
