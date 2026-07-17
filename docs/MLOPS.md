# EdgeSense AI — MLOps

How the anomaly model is versioned, monitored, swapped and promoted. This
chapter covers what **phase 1** ships — model manifest & versioning, serving-side
drift detection, hot reload, and a champion/challenger promotion gate — and
where the lifecycle goes next. Companion chapters: [`PLATFORM.md`](PLATFORM.md)
(tenancy, identity, registry), [`SECURITY.md`](SECURITY.md) (threat model —
§2.4 covers the ML-specific threats these controls start to address),
[`EVALUATION.md`](EVALUATION.md) (the quality bar itself).

## 1. The model lifecycle today

```
       make train / promote                     docker build            serve
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ synthesize   │──►│ fit 3→16→2→  │──►│ calibrate alarm  │──►│ bake bundle into │
│ healthy data │   │ 16→3 tanh AE │   │ threshold @99.5% │   │ inference image  │
│ (seeded)     │   │ (sklearn or  │   │ healthy quantile │   │ (or reload live) │
└──────────────┘   │  torch)      │   └──────────────────┘   └────────┬─────────┘
                   └──────────────┘            │                      │
                                               ▼                      ▼
                                    ┌──────────────────┐   ┌──────────────────┐
                                    │ offline eval:    │   │ POST /score for  │
                                    │ 25 episodes/fault│   │ the Go agent ·   │
                                    │ + 20k healthy    │   │ /metrics drift · │
                                    │ (evaluate.py)    │   │ /reload swap     │
                                    └──────────────────┘   └──────────────────┘
```

Training is deterministic given a seed: `ml/train.py` synthesizes healthy
operating data, fits the autoencoder, calibrates the alarm threshold on
held-out healthy data at the 99.5 % quantile, and emits a backend-agnostic
joblib bundle (raw numpy weights + scaler + threshold + z-guard). The
inference Docker image bakes a freshly trained model at build time;
`ml/model/*` is never committed.

## 2. What phase 1 adds

### 2.1 Model manifest, versioning & model card

Every trained bundle now carries a **manifest** (`ml/manifest.py`), embedded
in the bundle and written as a sidecar `model.manifest.json` next to it,
plus a generated human-readable `MODEL_CARD.md`:

- `schema_version` (1), `model_version`, `created_at`, `git_commit`
- **version scheme**: `{YYYYMMDD.HHMMSS}+{git7}` (UTC) — sortable, unique per
  retrain, traceable to a commit. Docker builds don't see `.git`, so the
  commit comes from the `EDGESENSE_GIT_COMMIT` build arg (falls back to
  `nogit`):
  `docker build --build-arg EDGESENSE_GIT_COMMIT=$(git rev-parse --short HEAD) …`
- training config: seed, epochs, architecture (layers + activation),
  false-positive budget, z-guard, sample counts
- training-data descriptor: generator id (`synthetic-normal-v1`), its
  distribution parameters, and a **sha256 of the training matrix** — the
  provenance record that SECURITY.md §2.4 asks for before field-data
  training lands
- metrics snapshot: calibrated threshold, validation FP/detection rates, and
  (when promoted through the gate) the offline evaluation numbers

Backward compatibility: the manifest is additive. `ml/scoring.py` ignores
unknown bundle keys and the server treats the manifest as optional, so
pre-phase-1 bundles keep loading (their version reports `unknown`).

```bash
make train                     # writes model.joblib + model.manifest.json + MODEL_CARD.md
curl -s localhost:8800/healthz # → …, "model_version": "20260717.203000+99c0e84", "created_at": …
```

### 2.2 Serving-side drift detection

The bundle already carries the training distribution (`scaler_mean` /
`scaler_scale`), so the sidecar can watch for drift without any new
artifacts (`inference/drift.py`): it keeps a rolling window (default 500
readings, `EDGESENSE_DRIFT_WINDOW`) of the raw readings it scores and
derives two per-feature signals:

- **z-shift** — the rolling mean's distance from the training mean in
  training standard deviations (signed; |z| ≳ 0.5 is a meaningful shift)
- **PSI** (population stability index) — the rolling distribution of the
  standardized feature vs the training distribution over fixed ±4σ bins
  (< 0.1 stable · 0.1–0.25 moderate · > 0.25 major drift)

Both are pure numpy over a small ring buffer — negligible CPU at 2 Hz. The
sidecar now serves Prometheus metrics on **`GET /metrics`** (port 8800):

| Metric | Type | Meaning |
|---|---|---|
| `edgesense_model_scored_total` | counter | readings scored by the sidecar |
| `edgesense_model_anomalies_total{reason}` | counter | flagged readings by trigger |
| `edgesense_model_score` | histogram | anomaly-score distribution |
| `edgesense_model_drift_zshift{feature}` | gauge | rolling-mean shift vs training (σ) |
| `edgesense_model_drift_psi{feature}` | gauge | PSI vs training distribution |
| `edgesense_model_drift_window_size` | gauge | readings in the drift window |
| `edgesense_model_info{model_version,kind,backend}` | gauge | live model metadata (=1) |
| `edgesense_model_reloads_total{result}` | counter | hot-reload attempts |

Prometheus scrapes the sidecar (`deploy/prometheus.yml`), the Grafana fleet
dashboard gains drift panels (PSI + z-shift per feature, live model version,
score p95), and a provisioned Grafana alert fires when any feature's PSI
stays above 0.2 for 10 minutes
(`deploy/grafana/provisioning/alerting/edgesense-drift.yml`).

Try it live: `make stack`, then inject a long fault episode
(`make demo` or the control topic) and watch the PSI panel react while the
alert counts down.

### 2.3 Hot model reload

The sidecar can swap models without a rebuild or restart:

```bash
python ml/train.py --out ml/model/model.joblib   # retrain in place
curl -X POST localhost:8800/reload
# → {"status":"reloaded","old_version":"…+99c0e84","new_version":"…+1a2b3c4", …}
```

- The candidate file is **validated before the swap** (feature list, scaler
  shapes, weight-chain dimensions, known activation, finite threshold, and a
  smoke score) — a missing, corrupt or malformed bundle returns **400** and
  the old model keeps serving.
- The swap is an atomic reference exchange: in-flight requests finish on the
  bundle they started with; scoring never sees a half-swapped model.
  (`ml/manifest.save_bundle` writes via temp-file + `os.replace`, so the
  file on disk is never half-written either.)
- On POSIX a **SIGHUP** triggers the same reload.
- `/healthz` reflects the live `model_version`; the drift window resets on
  swap (the new scaler defines a new reference frame).

### 2.4 Champion/challenger promotion gate

`ml/promote.py` is the gate every new model must pass before it replaces
the served bundle (`ml/model/model.joblib`, the **champion**):

```bash
make promote          # sklearn challenger (CPU)
make promote-torch    # PyTorch challenger (local only; CI stays CPU/sklearn)
python ml/promote.py --seed 43 --epochs 600   # args passthrough
```

1. trains a **challenger** (`--backend/--seed/--epochs`),
2. replays the offline evaluation harness (`ml/evaluate.py` — the exact
   simulator physics, 25 episodes per fault + 20 k healthy readings) on the
   challenger **and** the current champion,
3. checks the challenger against the **absolute quality bar** — every
   episode of every fault detected, ~1-reading median time-to-detect,
   healthy false-positive rate ≤ 0.6 % — and against **ONNX parity**
   (onnxruntime must reproduce the numpy scorer: relative score MAE < 1e-3,
   label agreement > 99 %),
4. compares challenger vs champion — episodes detected, per-fault median
   time-to-detect, healthy FP rate, within small tolerances — and then
   either **promotes** (atomically replaces `model.joblib` +
   `model.manifest.json` + `MODEL_CARD.md`) or **refuses** with a diff
   table. The champion is untouched on refusal.

Exit codes are CI-ready: `0` promoted · `1` refused · `2` error. The
candidate bundle, manifest, model card and gate report are always written
to `ml/model/candidate/` so they can be archived either way.

The **`model-gate`** workflow (`.github/workflows/model-gate.yml`,
`workflow_dispatch`) runs the same gate on a CPU runner and uploads the
candidate + report as a build artifact — models are never committed by CI,
and the regular test jobs are untouched.

## 3. Phase 2+ outlook

- **OTA model delivery to the edge** — ship promoted bundles to devices as
  signed artifacts (snap refresh or registry download), with signature
  verification before `/reload` per SECURITY.md §5/P4; the manifest's hash
  and version make the artifact verifiable end-to-end.
- **Shadow scoring** — serve the champion while a challenger scores the same
  readings in shadow; compare verdict streams on live traffic before
  promotion (the gate then adds online evidence to the offline bar).
- **Feedback loop & labeling** — capture operator confirm/dismiss verdicts
  on events, build a labeled corpus per machine, and feed it to evaluation
  (and eventually training) with the provenance recording the manifest
  already provides.
- **Per-machine thresholds** — calibrate the alarm threshold (and possibly
  the scaler) per machine once real fleets show per-asset baselines; the
  bundle format already carries both.
- **Model registry** — a registry service (PLATFORM.md §5) holding
  manifests, artifacts and promotion history per org/site, replacing
  bake-at-build with pull-by-version.
- **Continuous training** — scheduled retrain + gate runs once field data
  exists, with the poisoning mitigations of SECURITY.md §2.4 (authenticated
  sensor path, provenance, canary eval) as prerequisites.

## 4. Running everything

```bash
make train           # train + validate; writes bundle + manifest + model card
make eval            # offline evaluation -> docs/EVALUATION.md
make promote         # champion/challenger gate (see §2.4)
make export-onnx     # ONNX graph + sidecar metadata
make inference       # serve: POST /score · GET /healthz · GET /metrics · POST /reload
pytest tests/test_manifest.py tests/test_drift.py tests/test_reload.py tests/test_promote.py
```
