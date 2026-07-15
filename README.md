# EdgeSense AI

On-device anomaly detection for industrial machines:
sensors at the edge, ML inference on the node, and only *events* go upstream —
even when the uplink is down.

```
┌───────────┐ sensors  ┌────────────┐ /score  ┌────────────┐
│ simulator │ ───────► │ edge-agent │ ──────► │ inference  │
│ 3 machines│   MQTT   │    (Go)    │ ◄────── │ (FastAPI + │
└───────────┘  (local  │ disk buffer│   HTTP  │  autoenc.) │
      ▲        broker) └─────┬──────┘         └────────────┘
      │                      │ anomaly events only · QoS 1
      │ edgesense/control/   │ store-and-forward uplink
      │ fault (demos)        ▼
                      ┌──────────────┐       ┌────────────┐
                      │ cloud broker │ ────► │ dashboard  │
                      └──────────────┘       └────────────┘
   agent /metrics ──► Prometheus ──► Grafana (provisioned dashboard)
```

## Use cases

- **Predictive maintenance** — a worn bearing shows up as a 3–5× vibration
  multiple long before it seizes. EdgeSense raises the alarm on the **first
  faulty reading (~0.5 s at 2 Hz)** — see the demo below.
- **Thermal runaway protection** — overheat is a *single-feature* drift that
  isolation forests systematically missed (a failure mode found by this
  repo's own test suite); the autoencoder's reconstruction error flags it on
  the first reading, with the hybrid z-score guard as a certified hard
  backstop.
- **Overload / energy monitoring** — current excursions from jams or failing
  motors are flagged with a machine-readable `reason`, ready for maintenance
  ticketing.
- **Flaky uplinks (LTE, satellite, remote sites)** — detection never stops
  during an uplink outage; events buffer on disk and replay **exactly once,
  with original timestamps**, on reconnect.
- **Bandwidth economics** — raw telemetry stays on the node; only anomalies
  leave. At 2 Hz × 3 sensors a machine emits ~50 MB/day of raw data but
  typically **zero** upstream bytes on a healthy day.
- **Fleet operations** — every agent exports Prometheus metrics (readings/s,
  anomaly rates, buffer depth, uplink status, inference latency); a
  provisioned Grafana dashboard shows the whole fleet at a glance, and
  `make fleet MACHINES=25` scales the simulated plant on the spot.

## Quickstart (Docker, recommended)

```bash
make stack        # build + start both brokers, inference, agent, simulator, dashboard
make stack-logs   # follow logs
make smoke        # end-to-end check from the host (needs `make setup` once)
make stack-down   # stop everything
```

- Dashboard: http://localhost:8501 · Grafana: http://localhost:3000 (no login)
- Inference API: http://localhost:8800/healthz · Agent health/metrics: http://localhost:8890/healthz
- Prometheus: http://localhost:9090
- Local broker (sensors): localhost:11883 · Cloud broker (events): localhost:12883

Scale the simulated fleet without touching anything else:

```bash
make fleet MACHINES=25   # recreates the simulator with 25 virtual machines
```

The inference image bakes a freshly trained model at build time; the agent is
a distroless static Go binary with its event buffer on a named volume
(`agent-data`), so buffered events survive container restarts.

## Demos

Three scripted, self-verifying demos run against the live stack.

### 1. Fault injection with measured time-to-detect — `make demo`

Injects a bearing fault, an overheat and an overload into the running
simulator (MQTT control topic), then measures detection against ground truth:

```
--- scenario 1/3: bearing_fault on machine-01 ---
    worn bearing → vibration 3–5×, current +10–25%
    detected on reading #1 after 0.00s — 24/24 faulty readings flagged
    trigger reasons: limit×3, model+limit×21

=== summary ===
fault           machine       time-to-detect        flagged   reasons                  result
bearing_fault   machine-01    0.00s (reading #1)    24/24     limit×3, model+limit×21  PASS
overheat        machine-02    0.00s (reading #1)    24/24     limit×22, model+limit×2  PASS
overload        machine-03    0.00s (reading #1)    24/24     limit×7, model+limit×17  PASS

false positives: 0 across 171 healthy readings (0.0%; model is tuned for ~0.5%)
VERDICT: PASS
```

Inject your own faults while watching the dashboard:

```bash
mosquitto_pub -p 11883 -t edgesense/control/fault \
  -m '{"machine_id": "machine-02", "fault": "overheat", "ticks": 30}'
```

### 2. Uplink outage with zero event loss — `make demo-offline`

Kills the cloud broker, injects faults while it is down, restores it, and
verifies the buffered events (docker CLI required):

```
[1/4] baseline: uplink healthy — first event after 0.3s
[2/4] stopping cloud broker (edgesense-mosquitto-cloud) — uplink is now DOWN
      injected bearing_fault(machine-02) + overload(machine-03); events are buffering on the edge…
      agent metrics confirm: edgesense_buffer_depth = 36 events on disk
      events that reached the cloud during the outage: 0
[3/4] restoring cloud broker — ok
[4/4] replay: 35 buffered events delivered after restore (no duplicates)
      machines: machine-02, machine-03   original-reading age at delivery: 12.5s … 27.4s
      every event kept its original reading timestamp from inside the outage
      agent metrics confirm: edgesense_buffer_depth = 0 (buffer fully drained)

VERDICT: PASS — 35 events preserved across a 15.4s uplink outage
```

Watch it live in Grafana (http://localhost:3000): the uplink stat flips to
DOWN, the buffer-depth panel climbs during the outage and snaps back to zero
on replay.

### 3. Offline model evaluation — `make eval`

Replays the simulator's physics offline (25 episodes per fault, 20k healthy
readings) and writes [`docs/EVALUATION.md`](docs/EVALUATION.md):

| Fault | Episodes detected | Median time-to-detect | Reading recall |
|---|---|---|---|
| bearing_fault | 25/25 (100%) | 1 reading (~0.5 s) | 100% |
| overheat | 25/25 (100%) | 1 reading (~0.5 s) | 100% |
| overload | 25/25 (100%) | 1 reading (~0.5 s) | 100% |

False positives on healthy data: **0.43%** (the autoencoder's calibrated
false-alarm budget).

### 4. Public-dataset benchmark — `make benchmark`

Cross-checks the same architecture and calibration against real industrial
data: it trains on the healthy rows of the [AI4I 2020 Predictive Maintenance
dataset](https://archive.ics.uci.edu/dataset/601) (UCI, 10k milling readings,
labeled failure modes) and writes per-failure-mode recall and ROC-AUC to
[`docs/BENCHMARK.md`](docs/BENCHMARK.md). The CSV (~0.5 MB) is downloaded
once into `ml/data/` (gitignored); the pipeline itself is covered by an
offline test with synthetic data, so CI never touches the network.

## Components

| Path         | Role                                                              |
|--------------|-------------------------------------------------------------------|
| `simulator/` | Simulates machines publishing vibration / temperature / current over MQTT, with random or on-demand (control-topic) fault episodes. |
| `ml/`        | Trains a small autoencoder on normal operating data (sklearn or PyTorch backend) → `ml/model/model.joblib`; hybrid scoring (`scoring.py`); offline evaluation (`evaluate.py`); ONNX export (`export_onnx.py`). |
| `inference/` | FastAPI sidecar serving the model (`POST /score`).                 |
| `edge-agent/`| Go agent: subscribes to sensor topics, scores each reading, publishes anomaly events to the uplink broker with store-and-forward buffering. |
| `dashboard/` | Streamlit live dashboard: signals, anomaly markers, event feed.    |
| `scripts/`   | Self-verifying demos (`demo.py`, `demo_offline.py`) and smoke test. |
| `deploy/`    | Mosquitto config, Prometheus scrape config, Grafana provisioning (datasource + fleet dashboard). |
| `snap/`      | Snapcraft packaging for the edge agent (Ubuntu Core ready).        |

Every service ships a Dockerfile; `docker-compose.yml` wires them together on
an internal network with two brokers: `mosquitto` (local sensor bus) and
`mosquitto-cloud` (stand-in for a remote event broker).

## Quickstart (local processes)

Prerequisites:

- Python 3.12+
- Docker (for the brokers / full stack)
- Go 1.22+ — only for running the agent locally (`make agent`, `make test`); `make setup` skips the Go deps with a warning if Go is missing
- `mosquitto-clients` (optional, for manual fault injection)

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

Without `EDGESENSE_UPLINK_BROKER` set, the agent uses a single broker for
sensors and events — the demos and dashboard handle both layouts.

## Anomaly detection

The model is a small autoencoder (3 → 16 → 2 → 16 → 3, tanh) trained on
healthy operating data only. Healthy readings pass through the 2-unit
bottleneck almost unchanged; faults don't, so the mean squared reconstruction
error in scaled feature space is the anomaly score (**higher = more
anomalous**). The alarm threshold is calibrated on held-out healthy data at
the 99.5% quantile (~0.5% false-alarm budget). Two interchangeable training
backends emit the exact same bundle format — raw numpy weights, so inference
needs neither torch nor a fitted sklearn estimator:

```bash
make train                                      # sklearn MLPRegressor (default; CI + Docker)
.venv/bin/python ml/train.py --backend torch    # PyTorch (pip install -r requirements-torch.txt; CUDA if available)
.venv/bin/python ml/train.py --model iforest    # legacy IsolationForest baseline, for comparison
```

Scoring stays hybrid (`ml/scoring.py`): a reading is anomalous if the
reconstruction error exceeds the calibrated threshold **or** any feature
deviates more than `z_guard` (default 6σ) from the training distribution.
The guard is the certified hard limit for single-feature drift and a
backstop for anything the model might learn to reconstruct. Responses carry
a `reason` field: `model`, `limit`, or `model+limit`.

Compared head-to-head with the IsolationForest baseline (`ml/evaluate.py
--model <bundle>`), the autoencoder lifts model-side detection on synthetic
faults from ~61% to 100% of faulty readings at a slightly lower
false-positive rate — every alarm now carries the model's signature instead
of leaning on the limit guard. On real industrial data the same pipeline
separates labeled failure modes that per-feature limits cannot see at all —
see [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Store-and-forward

The agent publishes events with QoS 1 to the uplink broker. If the uplink is
unreachable, events are appended to a disk-backed FIFO (`EDGESENSE_BUFFER`,
JSON Lines, capped at 10k entries, oldest dropped first) and flushed on
reconnect plus every 30 s. The disk buffer is the single owner of offline
events (publishes are gated on a live connection), so replay is duplicate-free.
Events survive agent restarts. `make demo-offline` proves all of this live.

### Agent configuration

| Env var | Default | Purpose |
|---|---|---|
| `EDGESENSE_BROKER` | `tcp://localhost:11883` | local broker (sensor bus) |
| `EDGESENSE_UPLINK_BROKER` | = `EDGESENSE_BROKER` | broker events are published to |
| `EDGESENSE_INFERENCE_URL` | `http://localhost:8800/score` | scoring sidecar |
| `EDGESENSE_SENSOR_TOPIC` | `edgesense/sensors/#` | subscription filter |
| `EDGESENSE_BUFFER` | `event-buffer.jsonl` | store-and-forward file |
| `EDGESENSE_METRICS_ADDR` | `:8890` | Prometheus metrics + healthz listener |

## Observability

The agent exposes its operational state on `EDGESENSE_METRICS_ADDR`:

- `GET /healthz` → `{"status":"ok","uplink_connected":true,"buffer_depth":0}`
- `GET /metrics` → Prometheus text format

| Metric | Type | Meaning |
|---|---|---|
| `edgesense_readings_scored_total{machine}` | counter | readings scored per machine |
| `edgesense_score_failures_total` | counter | readings lost to inference errors |
| `edgesense_anomalies_total{machine,reason}` | counter | flagged readings by trigger |
| `edgesense_events_published_total` | counter | events delivered upstream (incl. replays) |
| `edgesense_events_buffered_total` | counter | events written to the disk buffer |
| `edgesense_buffer_depth` | gauge | events currently waiting on disk |
| `edgesense_uplink_connected` | gauge | 1 while the uplink connection is open |
| `edgesense_inference_latency_seconds` | histogram | scoring round-trip latency |

The compose stack ships Prometheus (scraping the agent every 5 s) and Grafana
with an auto-provisioned **EdgeSense AI — fleet & agent** dashboard
(anonymous access, no login): uplink status, buffer depth, per-machine
reading/anomaly rates, and p50/p95 inference latency. `make demo-offline`
asserts against these metrics — the buffer-depth gauge must drain to zero
after the replay.

## ONNX export

```bash
make export-onnx   # -> ml/model/model.onnx + model.onnx.json (scaler + guard + threshold)
```

The exported graph (~2 KiB) is self-contained: scaler, autoencoder weights
and the calibrated alarm threshold are baked in as constants — raw features
in, reconstruction-error score and 0/1 anomaly label out.
`tests/test_onnx.py` asserts parity between onnxruntime and the numpy scorer
(relative score MAE < 1e-3, label agreement > 99%). The exported model plus
the sidecar-free metadata file are the path to running inference directly in
the Go agent (roadmap).

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
make test         # pytest (model quality, API, simulator, evaluation, ONNX parity, benchmark pipeline, optional torch backend) + go test
```

GitHub Actions (`.github/workflows/ci.yml`) runs both suites on every push
and pull request: a Python 3.12 job (`pytest`) and a Go 1.22 job
(`go vet`, `go build`, `go test`).

## MQTT topics & ports

Host ports: local broker **11883**, cloud broker **12883** (1883 sits in a
Windows/Hyper-V reserved port range when running under WSL2 + Docker Desktop).

- `edgesense/sensors/<machine_id>` — raw readings (JSON), ~2 Hz per machine (local broker)
- `edgesense/events/<machine_id>` — anomaly events only (JSON, with score + reason; uplink broker)
- `edgesense/control/fault` — on-demand fault injection for demos (local broker)

## Ideas / roadmap

- [ ] Run ONNX inference inside the Go agent (onnxruntime bindings), drop the sidecar
- [x] Replace IsolationForest with an autoencoder for richer fault signatures
- [ ] CoAP uplink for constrained/LTE links
- [ ] Alerting: Grafana alert rules on buffer depth / uplink downtime
- [ ] Inference service as a second snap; model updates as snap refreshes

## References

- M. Feki, *Data Quality Model for Synthetic Image Data in Production*,
  Master's thesis, Technische Universität Berlin, Computer Vision & Remote
  Sensing — the author's related work on data quality for ML in production,
  which motivates the healthy-data-quality-first approach used here (train on
  verified healthy data only, calibrate the alarm budget on held-out data).
- S. Matzka, *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning
  Repository, 2020. <https://archive.ics.uci.edu/dataset/601> (CC BY 4.0) —
  used by `make benchmark` ([`docs/BENCHMARK.md`](docs/BENCHMARK.md)).
