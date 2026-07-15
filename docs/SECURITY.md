# EdgeSense AI — Security

**Security chapter: data, application, and device security**

| | |
|---|---|
| Status | **Assessment + proposal** — §1–§2 describe the code on `main` as it is; every fix is marked *(proposed)* and nothing in §3–§6 is implemented unless explicitly marked *current* |
| Scope | Current-state assessment, threat model (STRIDE + ML-specific), data security, application/supply-chain security, device security, phased hardening roadmap |
| Non-goals | Changing the detection pipeline; platform tenancy/RBAC design (that is [`PLATFORM.md`](PLATFORM.md) — this chapter reuses its identity phases rather than redesigning them) |
| Companions | [`PLATFORM.md`](PLATFORM.md) (device identity §4, roadmap §7) · [`GLOSSARY.md`](GLOSSARY.md#security) (security terms) |

---

## 1. Security posture today (honest assessment)

Everything in this section is grounded in the code on `main`; file references point at
the source of truth. The short version: **the architecture minimizes exposure by
design, but nothing is authenticated and nothing is encrypted.** The demo threat model
is "trusted lab network", and the code is honest about it — the gaps below are absences,
not bugs.

### 1.1 What exists and is worth keeping

- **Events-only uplink (data minimization).** Raw telemetry (~50 MB/day/machine) never
  leaves the node; only anomaly events cross the WAN boundary
  (`edge-agent/main.go`, README architecture). The most sensitive data class has the
  smallest exposure by construction.
- **Distroless static Go services.** The agent and the CoAP receiver are
  `CGO_ENABLED=0` static Go binaries on `gcr.io/distroless/static-debian12`
  (`edge-agent/Dockerfile`, `coap-receiver/Dockerfile`): no shell, no package
  manager, no libc — a minimal exploitation and CVE surface.
- **Strict snap confinement.** The production packaging declares
  `confinement: strict` with only the `network` plug (`snap/snapcraft.yaml`): AppArmor/
  seccomp sandboxing, no filesystem access beyond snap-owned paths, and snapd's signed,
  auto-refreshing update channel.
- **No secrets in the repo.** No credentials, tokens, or keys exist in code, compose
  files, or CI — verified; though today that is because *nothing requires
  authentication* (see §1.2), it also means phase 0 starts from a clean slate.
- **Input validation at the boundaries that exist.** The inference API validates types
  via pydantic (`inference/server.py`); the CoAP receiver rejects `machine_id` values
  containing `/ + # \0` before building an MQTT topic — a real topic-injection guard
  (`validate`, `coap-receiver/handler.go`); the agent drops unparseable payloads
  (`edge-agent/main.go`).
- **At-least-once delivery with an atomic buffer.** QoS 1 publishes plus a
  crash-safe (temp-file + rename) disk buffer (`edge-agent/buffer.go`) — an
  *availability* control: an uplink outage is not an event-loss incident.
- **Hash-pinned Go dependencies.** `go.sum` verifies module content against the Go
  checksum database in both modules — supply-chain integrity Go gives for free.
- **No broker persistence.** `persistence false` (`deploy/mosquitto.conf`) — no
  at-rest message queue on either broker to protect or wipe.
- **Deterministic in-build model training.** The model is trained on synthetic data at
  image build time (`inference/Dockerfile` runs `ml/train.py`); there is no field-data
  training path an attacker could poison today (§2.4).

### 1.2 Surface → risk matrix

Every row is verified against the file in the *current state* column. Targets reference
the roadmap phases in §6; all targets are *(proposed)*.

| # | Surface | Current state | Risk | Target (phase) |
|---|---|---|---|---|
| S1 | Local broker (host port 11883) | `allow_anonymous true`, plaintext, no ACLs (`deploy/mosquitto.conf`); port published to the host, i.e. the plant LAN | Anyone on the network reads raw telemetry, spoofs sensor readings, injects faults | Auth + ACLs; bind node-local (P0) |
| S2 | Cloud broker (host port 12883) | Same anonymous config — the *same file* is mounted into both brokers (`docker-compose.yml`) | Forged anomaly events, eavesdropping on plant-health data crossing the WAN | TLS 8883 + per-device creds (P0) → mTLS (P3) |
| S3 | `edgesense/control/fault` topic | Open to any client; README documents unauthenticated `mosquitto_pub` fault injection | Anyone on the network drives fault injection: ops disruption, alert fatigue, masking of real faults | ACL: control topic writable by ops role only (P0) |
| S4 | CoAP uplink (udp 15683 → 5683) | NoSec mode, plaintext CBOR, no client auth; `coaps://` explicitly rejected as unsupported (`edge-agent/uplink.go`) | Event forgery/eavesdropping on the constrained link; UDP source spoofing | DTLS-PSK, later OSCORE (P5) |
| S5 | Inference API (host port 8800) | Plain HTTP, no auth (`inference/server.py`, `docker-compose.yml`) | Open scoring oracle for evasion tuning (§2.4); DoS on the scoring path | Internal-only + authn (P0/P2) |
| S6 | Model artifact | `joblib.load` of a pickle baked into the image (`inference/server.py`, `EDGESENSE_MODEL` override) | Artifact substitution = arbitrary code execution in the inference container | Signed artifacts; ONNX-only load path (P4) |
| S7 | Event buffer file | JSONL via `os.Create` → default 0644, plaintext, no integrity check; unparseable lines silently dropped (`edge-agent/buffer.go`) | Local tamper forges "authentic" QoS 1 events upstream or silently destroys evidence (§2.4) | 0600 perms; per-line MAC (P2) |
| S8 | Grafana (host port 3000) | Anonymous access enabled with org role **Admin**, login form disabled (`docker-compose.yml`) | Anyone on the network administers dashboards/datasources; pivot to Prometheus | Anonymous off (P0); OIDC per `PLATFORM.md` phase 4 |
| S9 | Streamlit dashboard (8501) | No login (`dashboard/app.py`) | Full fleet event visibility to anyone on the network | Reverse-proxy auth (P2) |
| S10 | Prometheus (9090) / agent metrics (8890) / CoAP receiver metrics (8891) | No auth, host-published | Fleet reconnaissance (anomaly rates, buffer depth, uplink state); PromQL resource abuse | Internal network only (P2) |
| S11 | MCP server (host port 8900) | Streamable HTTP on `0.0.0.0`, no auth, exposes an `inject_fault` tool (`mcp_server/server.py`, `mcp_server/Dockerfile`) | Unauthenticated HTTP→MQTT control bridge: fault injection for anyone who can reach the port | Localhost/stdio default; token auth (P2) |
| S12 | CI pipeline | Actions pinned by tag (`@v4`/`@v5`), no dependency/container scanning, no SBOM, no dependabot config (`.github/workflows/ci.yml`) | Supply-chain drift; vulnerable dependencies ship silently | SHA pins, scanners, SBOM, dependabot (P1) |
| S13 | Python dependencies | `requirements*.txt` unpinned (floors only) | Non-reproducible builds; silent upgrades into vulnerable/hijacked releases | Compiled, hash-pinned requirements (P1) |
| S14 | Containers | All services run as root (no `USER` in Python images; distroless default user is root — a `:nonroot` tag exists); one flat compose network | Larger blast radius after any single-service compromise; free lateral movement | Non-root users; segmented networks (P2) |
| S15 | Snap packaging | `confinement: strict`, `network` plug only — but `grade: devel` (`snap/snapcraft.yaml`) | Mostly a *strength*; devel grade blocks stable-channel publication | Keep; `grade: stable` at release (P3) |

### 1.3 Trust boundaries

```
        PLANT NETWORK (untrusted: any employee/visitor/compromised host)
  ─────────────────────────────────────────────────────────────────────────
   host-published today: 11883 12883 8800 8890 8501 8900 9090 3000
                         + coap profile: 15683/udp 8891
  ═════════════╦═══════════════════════════════════════════╦═══════════════
               ║  TB1: edge device boundary                ║
   ┌───────────╨───────────────────────────────┐           ║
   │ EDGE DEVICE (snap / compose host)         │           ║
   │  sensors ─► local broker ─► agent ─► inference        ║
   │  (S1: anonymous)   │   (S5: open HTTP)    │           ║
   │  control topic ────┘ (S3: open)           │           ║
   │  buffer.jsonl (S7: 0644, no MAC)          │           ║
   └───────────┬───────────────────────────────┘           ║
               │ TB2: WAN uplink (LTE/satellite —          ║
               │      plaintext MQTT or NoSec CoAP: S2/S4) ║
   ┌───────────▼───────────────────────────────┐           ║
   │ CLOUD / UPSTREAM                          │           ║
   │  coap-receiver ─► cloud broker ─► dashboard (S9)      ║
   │  Prometheus (S10) ─► Grafana (S8: anon Admin)         ║
   │  MCP server (S11: HTTP control bridge)    │           ║
   └───────────┬───────────────────────────────┘           ║
               │ TB3: operator access (browser/MCP client) ║
  ═════════════╩═══════════════════════════════════════════╩═══════════════
   Today every boundary is crossable without credentials; TB2 additionally
   carries plant-health data in plaintext. The demo collapses all three
   zones onto one Docker host — in production TB1 is one box per machine.
```

---

## 2. Threat model

### 2.1 Attacker personas

| Persona | Access | Representative goals |
|---|---|---|
| **On-network attacker** | Plant LAN / Wi-Fi; can reach every host-published port in §1.3 | Read process telemetry (trade secrets), inject faults (S3), forge or suppress anomaly events, deface dashboards |
| **Compromised device** | Full control of one edge node (agent, buffer, local broker, keys once they exist) | Publish as other machines, poison future training data, tamper the buffer, pivot upstream over the uplink |
| **Malicious insider** | Legitimate operator/developer credentials; repo or dashboard access | Silence alarms before sabotage, exfiltrate fleet health, slip a change into CI |
| **Supply-chain attacker** | Controls an upstream dependency, base image, action, or model artifact | Execute code in CI or in containers (S6, S12, S13), backdoor the agent binary |

### 2.2 Dataflow under analysis

Numbered hops; the STRIDE table below keys on them.

```
 sensors ──F1──► local broker ──F2──► edge agent ──F3──► inference (/score)
 (simulator)     (mosquitto)             │  ▲                (FastAPI)
      ▲                                  │  └────────────────────┘
      └──F7── edgesense/control/fault    │
              (any client, README demo)  │ F4a: MQTT QoS 1 ─────► cloud broker
              MCP inject_fault ──F8──────┤ F4b: CoAP CON/UDP ──► coap-receiver
                                         │                        └─F4c─► cloud broker
                            buffer.jsonl◄┘ (on uplink failure)
 cloud broker ──F5──► dashboard (Streamlit)
 agent /metrics ──F6──► Prometheus ──F6──► Grafana
```

### 2.3 STRIDE analysis

| STRIDE | Concrete threat (flow) | Enabling weakness (today) | Mitigation *(proposed)* |
|---|---|---|---|
| **S**poofing | Publish readings or events as any machine (F1, F4a): `machine_id` is free-form payload data with a topic-segment fallback (`topicMachineID`, `edge-agent/main.go`) | Anonymous brokers, no device identity | Per-device credentials + ACLs (P0), mTLS with CN = `device_uid` (P3, `PLATFORM.md` §4.3) |
| | Spoofed-source CoAP POSTs (F4b) — UDP, NoSec | No DTLS/OSCORE | DTLS-PSK per device (P5) |
| **T**ampering | Edit/delete/insert events in `buffer.jsonl` before replay (§2.4) | 0644 plaintext file, no MAC; forged lines replay upstream as QoS 1 events | 0600 + per-line MAC keyed by device secret (P2/P3) |
| | Modify events or readings in flight (F2, F4a/b/c) | Plaintext MQTT and CoAP everywhere | TLS on uplink (P0), mTLS (P3), DTLS (P5) |
| | Swap the model artifact (F3) | Unsigned joblib pickle; `EDGESENSE_MODEL` env override | Signed artifacts, ONNX-only loading (P4) |
| **R**epudiation | Forged events/faults are unattributable: client IDs are fixed strings, not identities (`PLATFORM.md` §1) | No authentication → no meaningful audit trail; broker logs to stdout only | Device identity (P0/P3) + retained broker logs with authenticated client IDs (P2) |
| **I**nformation disclosure | Read raw telemetry on the LAN (F1/F2) — a ~50 MB/day/machine process signature | S1 anonymous + host-published | Auth + node-local binding (P0) |
| | Eavesdrop plant-health events on the WAN (F4) | No TLS | TLS 8883 (P0) |
| | Fleet recon via open Grafana/Prometheus/dashboard/metrics (F5, F6) | S8–S10 | Anonymous off (P0), internal network + auth (P2) |
| | Model extraction via the open `/score` oracle (F3) | S5 | Internal-only + authn (P0/P2), rate limits |
| **D**enial of service | Anomaly-flood eviction: buffer caps at 10 000, **oldest dropped first** (`edge-agent/buffer.go`) — flood fake anomalies during an outage to evict real buffered events | S1 lets anyone publish readings that score anomalous | Broker auth (P0); alert on `edgesense_buffer_depth` spikes (metric already exists — *current*) |
| | Connection/publish floods on brokers; UDP floods on the CoAP receiver; scoring-path saturation (F3) | No auth, no rate limits | Auth (P0), per-client broker limits, receiver rate limiting (P5) |
| **E**levation of privilege | Anonymous Grafana **Admin** (S8): datasource/dashboard admin for anyone | `GF_AUTH_ANONYMOUS_ORG_ROLE: Admin` | Anonymous off (P0), OIDC (`PLATFORM.md` phase 4) |
| | `inject_fault` via unauthenticated MCP HTTP (F8/S11) | MCP binds `0.0.0.0:8900`, no auth | stdio/localhost default + token (P2) |
| | Pickle deserialization = code execution in the inference container (S6); root containers ease post-exploit movement (S14) | joblib + root + flat network | ONNX (P4), non-root + segmentation (P2) |

### 2.4 ML-specific threats

**Training-data poisoning.** Today the model trains on *synthetic* healthy data
generated deterministically at image build (`ml/train.py` via `inference/Dockerfile`)
— there is no field-data path to poison, so the current poisoning surface is the
*supply chain* (repo + CI + base images), covered by §4. The moment
retraining-on-field-telemetry lands (a natural roadmap item), the anonymous local
broker becomes a poisoning channel: an attacker publishes abnormal-but-labeled-healthy
readings so the retrained model learns to accept a developing fault, and the
recalibrated 99.5 % threshold (`FP_BUDGET`, `ml/train.py`) drifts upward.
Mitigations *(proposed)*: authenticate the sensor path *before* any field-data
training (P0 is a prerequisite), record data provenance per training batch, gate
retrained models on a held-out canary suite (the evaluation harness in
`ml/evaluate.py` already exists — *current*), and alert on threshold drift between
model versions.

**Evasion (crafting readings under the alarm).** Detection is a hybrid: reconstruction
error above a calibrated threshold OR any feature beyond the 6σ z-guard
(`ml/scoring.py`). The evasion window is a fault signature that stays under *both* —
e.g. slow single-feature drift held below 6σ that the autoencoder reconstructs
acceptably. Two things make this cheaper today: the open `/score` endpoint (S5) is a
perfect offline oracle for tuning payloads before touching the plant, and the open
local broker (S1) lets an attacker inject the tuned readings as any machine. Note the
limit of the threat: a *network* attacker can add fake readings but cannot delete the
real sensor stream — suppressing a real fault requires compromising the device or the
sensor path itself (persona 2). Mitigations *(proposed)*: authenticate `/score` and
the sensor topic (P0/P2), rate-limit scoring, monitor the `reason` mix
(`edgesense_anomalies_total` labels — *current*): a fleet that suddenly only ever
triggers `limit` and never `model` deserves suspicion.

**Model theft.** The `/score` oracle enables decision-boundary extraction, and the
artifact itself ships inside the inference image (S6). Honest assessment: today's
model is trained on synthetic data and has near-zero confidentiality value — the risk
becomes real when models are trained per customer on proprietary process data.
Mitigations *(proposed)*: authn + rate limits on `/score` (P2), registry access
control on images, signed/encrypted model artifacts (P4).

**The store-and-forward buffer as a tamper target.** `event-buffer.jsonl` is the one
place where *future upstream truth* sits on disk unprotected (S7): world-readable
(0644 via `os.Create`), unauthenticated content, and lines that fail to parse are
*silently skipped* on read (`edge-agent/buffer.go`) — so tampering is also quiet
destruction. An attacker with file access can forge events that the agent will
faithfully replay upstream with QoS 1 and original timestamps, delete the evidence of
an outage window, or corrupt lines to the same effect. In the snap the file lives in
root-owned `$SNAP_COMMON` (other strict snaps cannot reach it — *current* strength);
in compose it lives in the `agent-data` named volume. Mitigations *(proposed)*: create
with 0600 (one-line change in `buffer.go`), append a per-line MAC keyed by a device
secret once device identity exists (P3), alert on `edgesense_buffer_depth` moving
without a corresponding uplink-down period (both metrics exist — *current*).

---

## 3. Data security

### 3.1 Classification

| Class | Examples | Where it lives today | Sensitivity | Current protection → target |
|---|---|---|---|---|
| Raw telemetry | 2 Hz vibration/temperature/current readings | Local broker only (never uplinked — *current*, by design) | **Moderate–high**: continuous process signature = trade secret | None on the LAN hop → node-local broker + auth (P0) |
| Anomaly events | `{machine_id, ts, score, reason, reading, agent_ts}` | Uplink brokers, buffer file, dashboard | **High**: plant-health and downtime posture; crosses the WAN | Plaintext → TLS (P0), mTLS (P3), buffer MAC (P2/P3) |
| Model artifacts | `model.joblib`, ONNX exports | Baked into the inference image | **Integrity-critical** (pickle = code, S6); confidentiality low today | Unsigned → signed + ONNX-only (P4) |
| Credentials & keys | *(none exist today — verified)*; future broker passwords, bootstrap tokens, device keys, CA | — | **Secret** | n/a → docker secrets / sops, TPM-backed keys (P0, P3, §5) |
| Operational metrics | Prometheus series, `/healthz` | Prometheus, Grafana, agent endpoints | **Low–moderate**: fleet recon value | Open → internal network + auth (P0/P2) |

### 3.2 Data in transit *(all proposed)*

| Link | Today | Target |
|---|---|---|
| Agent → cloud broker (MQTT) | `tcp://` plaintext | **P0:** TLS on 8883, server cert; per-device username/password (`PLATFORM.md` §4.2). **P3:** mTLS, CN = `device_uid` (`PLATFORM.md` §4.3) |
| Agent → CoAP receiver | NoSec UDP 5683, CBOR | **P5:** DTLS-PSK (`coaps://`, 5684); OSCORE as follow-up (below) |
| Sensors → local broker | Plaintext on the node/LAN | **P0:** bind node-local + auth; TLS optional on-node (loopback) |
| Agent → inference | Plain HTTP on the compose network / localhost | **P2:** stop publishing 8800 to the host; internal network + token; TLS if it ever leaves the node |
| Browsers → Grafana/dashboard | Plain HTTP | **P2:** reverse proxy with TLS + auth |

Two facts make the MQTT-TLS step cheaper than it looks *(current)*: the agent's uplink
already dispatches on URL scheme and hands anything non-CoAP to paho — including
`ssl://` (`edge-agent/uplink.go`) — and the distroless static base image ships CA
roots. Server-authenticated TLS to a broker with a publicly trusted certificate is
nearly configuration-only; a private CA or client certificates need a new config
surface (there are no `EDGESENSE_TLS_*` variables today).

Broker-side sketch *(proposed — this file does not exist; today one shared
`deploy/mosquitto.conf` serves both brokers)*:

```
# deploy/mosquitto-cloud.conf (proposed, P0)
listener 8883
cafile   /mosquitto/certs/ca.crt
certfile /mosquitto/certs/server.crt
keyfile  /mosquitto/certs/server.key
allow_anonymous false
password_file /mosquitto/config/passwd
acl_file      /mosquitto/config/acl
```

```
# deploy/mosquitto-acl (proposed, P0) — quick-win ACL, flat topics
user agent
topic write edgesense/events/#
user dashboard
topic read edgesense/events/#
user ops
topic write edgesense/control/fault
user simulator
topic read edgesense/control/fault
topic write edgesense/sensors/#
# Phase 1 of PLATFORM.md §4.2 replaces this with pattern rules on es/%u/…
```

**CoAP on constrained links — DTLS vs OSCORE.** DTLS 1.2 gives channel security with
an existing Go implementation (pion/dtls, integrated in the go-coap stack), PSK mode
avoids certificate weight on NB-IoT/LTE-M, and Connection IDs (RFC 9146) reduce
re-handshakes on flaky links — but a handshake per "connection" is still real cost on
a link that drops constantly, and DTLS terminates at the receiver (hop-by-hop).
OSCORE (RFC 8613) protects each request/response at the CoAP layer end-to-end,
survives proxies, needs no handshake on reconnect (sequence numbers only) — the
better long-term fit for exactly this store-and-forward pattern — but mature Go
library support is lacking and key establishment (EDHOC, RFC 9528) is young.
Pragmatic order *(proposed)*: DTLS-PSK first (the `coaps://` seam already exists in
`uplink.go` — it currently returns "not supported yet"), track OSCORE.

### 3.3 Data at rest *(all proposed unless marked)*

- **Buffer file permissions:** create with `0600` (replace `os.Create` with
  `os.OpenFile(..., 0o600)` in `edge-agent/buffer.go`; also tighten `MkdirAll` from
  `0755`). *Current:* atomic replace already prevents corruption from crashes.
- **Buffer integrity:** per-line MAC keyed by a device secret once one exists (P3
  dependency — there is no device key before then), turning silent tamper (§2.4) into
  a detectable event; alert on parse/MAC failures instead of skipping quietly.
- **Model artifacts:** sign at training time (cosign/minisign detached signature),
  verify at load in `inference/server.py`; prefer the ONNX path (`ml/export_onnx.py`
  exists — *current*) so loading a model never executes pickle bytecode (P4).
- **Broker state:** `persistence false` — nothing to protect *(current)*; revisit if
  persistence is ever enabled.
- **Snap:** buffer lives in root-owned `$SNAP_COMMON`, unreachable by other strict
  snaps *(current)*; full-disk encryption comes with Ubuntu Core + TPM on real
  hardware (§5).

---

## 4. Application security *(all proposed)*

Ordered by effort-to-value; everything here is CI/config, no service code.

| Control | Tool | Where it lands |
|---|---|---|
| Dependency update PRs | dependabot (`gomod` ×2, `pip` ×5, `docker` ×6, `github-actions`) | `.github/dependabot.yml` (new) |
| Python vuln scan | `pip-audit` job | `.github/workflows/ci.yml` |
| Go vuln scan | `govulncheck ./...` in the module matrix | `.github/workflows/ci.yml` |
| Image scan | trivy (or grype) on the six built images | `.github/workflows/ci.yml` |
| SBOM | syft → SPDX/CycloneDX artifact per image, attached to releases | `.github/workflows/ci.yml` |
| Reproducible Python deps | `pip-compile --generate-hashes`; commit lockfiles | `requirements*.txt`, per-service `requirements.txt` |
| Action pinning | Pin `actions/*` by commit SHA (tags are movable): `actions/checkout@<sha> # v4` | `.github/workflows/ci.yml` |
| Least-privilege workflow token | `permissions: contents: read` at workflow level | `.github/workflows/ci.yml` |
| Base image pinning | Pin `python:3.12-slim`, `eclipse-mosquitto:2`, etc. by digest (Prometheus/Grafana are already version-tag pinned — *current*) | all Dockerfiles, `docker-compose.yml` |
| Non-root containers | `gcr.io/distroless/static-debian12:nonroot` for the agent and receiver; `USER` + venv-in-`/app` for the four Python images | 6 Dockerfiles |
| Network segmentation | Compose networks: `sensors` (broker+simulator+agent), `uplink` (agent+cloud broker+receiver), `obs` (metrics stack); stop publishing 8800/8900/9090 to the host | `docker-compose.yml` |
| Secrets management | *Current:* zero secrets in the repo — keep it true when P0 credentials arrive: docker secrets (file mounts) or sops-age encrypted env, never plaintext env in compose; snap side via a root-owned 0600 env file | `docker-compose.yml`, deploy docs |

Supply-chain notes grounded in today's files: Go modules are already hash-verified
(`go.sum` + sumdb — *current*); the Python side is the soft spot (S13, unpinned) and
the CI actions are tag-pinned (S12). The inference image both trains and serves —
after P4, CI should also record model provenance (git SHA + training data hash) next
to the SBOM.

---

## 5. Device security

The identity design lives in [`PLATFORM.md`](PLATFORM.md) §4; this section covers what
surrounds it on the physical device. Everything here is *(proposed)* except where
marked.

- **Identity lifecycle** — adopt `PLATFORM.md` §4.3 as-is: one-time bootstrap token →
  keypair generated on-device (private key never leaves) → CSR with CN = `device_uid`
  → registry-signed ~90-day cert → EST-style renewal over the existing mTLS channel →
  revocation = registry disable in dynsec + short lifetimes (CRLs as backstop). The
  agent's store-and-forward already makes a botched renewal a buffering event, not
  data loss *(current)*.
- **Key storage: secure element / TPM.** Where hardware allows, generate and keep the
  device key in a TPM 2.0 or secure element (e.g. ATECC608 on Pi-class gateways) and
  expose it to the agent as a `crypto.Signer` (PKCS#11 / go-tpm) so mTLS private keys
  are non-exportable; fall back to a 0600 file keystore on hardware without one.
  Document per-SKU in provisioning docs.
- **Secure boot** (real hardware note): UEFI Secure Boot or U-Boot verified boot (FIT
  signatures) so only signed OS/firmware runs; on Ubuntu Core, secure boot + TPM also
  unlock full-disk encryption. Measured boot binding cert issuance to firmware state
  is the advanced tier — out of scope until there is real fleet hardware.
- **Snap strict confinement is the update + sandbox story** *(current strength)*:
  snapd verifies snap signatures against store assertions, auto-refreshes with
  automatic rollback on failed health, channels give staged rollout
  (edge → candidate → stable), and strict confinement (only the `network` plug —
  `snap/snapcraft.yaml`) bounds a compromised agent's blast radius. Change needed:
  `grade: devel` → `grade: stable` to publish beyond edge/beta channels.
- **Rotation & revocation:** short-lived certs make rotation routine (renewal
  window ≫ outage tolerance thanks to the buffer); revocation is registry-driven
  dynsec disable — takes effect at the broker without touching the device
  (`PLATFORM.md` §4.3).
- **Decommissioning checklist:** (1) revoke: dynsec disable + registry release
  (`RELEASED` state, `PLATFORM.md` §2.2); (2) wipe: buffer file and credentials —
  `$SNAP_COMMON/event-buffer.jsonl` (snap) or the `agent-data` volume (compose), plus
  the keystore if not hardware-bound; (3) verify: broker rejects the old identity;
  (4) re-claim requires a fresh claim code — never reuse bootstrap tokens (single-use
  by design).

---

## 6. Hardening roadmap *(all proposed)*

Phased like `PLATFORM.md` §7: each phase ships independently and is useful alone.
Quick wins first; `make stack` demo ergonomics survive every phase (credentialed
defaults generated at `make setup` time, or a documented `--insecure` demo profile).
Phases P3+ deliberately reuse `PLATFORM.md` roadmap machinery instead of duplicating
it.

| Phase | Delivers | Touches (files) | Depends on | Done when |
|---|---|---|---|---|
| **P0 — Broker auth + control ACL + uplink TLS** (quick wins) | Split broker configs; `allow_anonymous false` + `password_file` + ACL on both brokers (control topic writable only by `ops`); TLS (server cert) on the cloud broker's 8883; agent connects with `ssl://` + credential env vars; Grafana anonymous **off**; stop publishing internal-only ports (8800, 8900, 9090) to the host | `deploy/mosquitto.conf` → `deploy/mosquitto-local.conf` + `deploy/mosquitto-cloud.conf` + `deploy/mosquitto-acl` + passwd; `docker-compose.yml`; `edge-agent/main.go`, `edge-agent/uplink.go` (creds/TLS options); `simulator/simulate.py`, `dashboard/app.py`, `mcp_server/server.py`, `scripts/demo*.py` (creds); README | — | Anonymous connects are refused on both brokers; an unauthenticated `mosquitto_pub` to `edgesense/control/fault` fails; `make demo` passes with credentials |
| **P1 — CI supply chain** | dependabot; `pip-audit` + `govulncheck` + trivy jobs; SBOM per image; SHA-pinned actions; `permissions: contents: read`; hash-pinned requirements | `.github/dependabot.yml` (new); `.github/workflows/ci.yml`; `requirements*.txt`, `dashboard/requirements.txt`, `inference/requirements.txt`, `simulator/requirements.txt`, `mcp_server/requirements.txt` | — | CI fails on a known-vulnerable dependency; SBOMs published as artifacts; all actions SHA-pinned |
| **P2 — Runtime hardening** | Non-root containers; compose network segmentation; buffer file 0600; MCP behind localhost/token; auth proxy for dashboard/Grafana/Prometheus; retained broker logs | 6 Dockerfiles; `docker-compose.yml`; `edge-agent/buffer.go` (perms); `mcp_server/server.py`; `deploy/` proxy config (new) | P0 | No container runs as uid 0; inference/MCP/Prometheus unreachable from the host network; buffer unreadable by non-owner |
| **P3 — mTLS device identity** | Platform CA + registry-signed short-lived certs, `require_certificate` on the uplink listener, EST-style renewal, dynsec revocation; buffer per-line MAC keyed by the device key; `grade: stable` | = `PLATFORM.md` phases 2–3: registry (new service); `deploy/mosquitto-cloud.conf`; `edge-agent/uplink.go` (TLS client config), `edge-agent/buffer.go` (MAC); `snap/snapcraft.yaml` | P0 (P2 recommended) | Password auth off on the uplink listener; revocation takes effect without touching the device; tampered buffer lines are detected, not skipped |
| **P4 — Signed models, pickle-free inference** | Sign model artifacts at training; verify at load; serve from ONNX only (no `joblib.load` in the serving path); record model provenance in CI | `ml/train.py`, `ml/export_onnx.py`; `inference/server.py`, `inference/Dockerfile`; `.github/workflows/ci.yml` | P1 | Inference refuses an unsigned/modified artifact; no pickle deserialization at serve time |
| **P5 — CoAP link security** | `coaps://` DTLS-PSK uplink (per-device PSK from the registry) with Connection IDs for flaky links; receiver rate limiting; OSCORE evaluated when Go support matures | `edge-agent/coap.go`, `edge-agent/uplink.go` (replace the `coaps` "not supported" stub); `coap-receiver/main.go`; `docker-compose.yml` | P3 (registry-issued PSKs) | NoSec CoAP refused by the receiver; `make demo-offline-coap` passes over DTLS |

Sequencing rationale: P0 closes the "anyone on the network owns the plant" class (S1–S3,
S8) with configuration only; P1 is independent and cheap, so it runs in parallel; P2
contains the blast radius of whatever gets through; P3 makes identity cryptographic and
unlocks integrity features that need a device key (buffer MAC); P4–P5 retire the two
remaining specialized surfaces (pickle, NoSec UDP).

---

## 7. Accepted demo risks

For the single-host Docker demo on a private lab network, the following remain
acceptable *by explicit choice, documented here*: the simulator publishes without
hardware sensor identity (it *is* the spoofing tool); the local broker stays reachable
by demo scripts on the host; Streamlit remains loginless until the platform dashboard
work (`PLATFORM.md` phase 4). What is **not** acceptable even for a demo on a shared
network — and is why P0 exists — is the open control topic, anonymous Grafana Admin,
and an unauthenticated HTTP bridge (MCP) that can inject faults.
