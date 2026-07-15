# EdgeSense AI — Glossary

The definitions book for the repo, the platform design
([`PLATFORM.md`](PLATFORM.md)), the security chapter
([`SECURITY.md`](SECURITY.md)) and the hardware chapter
([`HARDWARE.md`](HARDWARE.md)). Definitions of **current** behavior are grounded in the
code (file references given). Terms introduced by these design chapters are marked
***(proposed)*** — they describe intent, not shipped behavior.

Groups: [Domain & telemetry](#domain--telemetry) ·
[Detection](#detection) ·
[Transport & reliability](#transport--reliability) ·
[Platform & identity](#platform--identity) ·
[Security](#security) ·
[Hardware](#hardware)

---

## Domain & telemetry

**Machine** — the monitored physical asset (pump, press, motor). Identified by
`machine_id`, a free-form string taken from the reading payload, with the machine
topic segment as fallback (`machineIDFromTopic`, `edge-agent/topics.go`: 4th segment
under `es/<org>/<site>/<machine>/…`, last segment in the legacy layout). The unit of
scoring, alerting, and per-machine metrics.

**Device** — the compute endpoint running the edge agent. In this codebase Machine and
Device are 1:1 (the agent ships as one snap daemon per node, `snap/snapcraft.yaml`), so
the platform design enrolls them as a single object with a stable `device_uid` *(the
UUID part is proposed; today the only identity is the free-form `machine_id`)*.

**Sensor** — one measured channel of a machine. Exactly three today: `vibration`
(mm/s RMS), `temperature` (°C), `current` (A) — the model's feature vector
(`FEATURES`, `ml/train.py`; units in `simulator/simulate.py`).

**Reading** — one JSON sample of all sensors of one machine at one instant:
`{machine_id, ts, vibration, temperature, current}`, published at 2 Hz (0.5 s interval)
to `edgesense/sensors/<machine_id>` on the local broker (`simulator/simulate.py`,
`reading` struct in `edge-agent/main.go`). Raw readings stay on the node.

**Fault** — an abnormal machine condition. The simulator produces three, with distinct
physical signatures (`simulator/simulate.py`):
- **`bearing_fault`** — worn bearing: vibration ×3–5, current ×1.1–1.25.
- **`overheat`** — thermal runaway: temperature ramps from +15 °C to +30 °C over the
  episode; a *single-feature* drift (the case that motivated the z-guard and the
  autoencoder — see README "Thermal runaway protection").
- **`overload`** — jam / failing motor: current ×1.6–2.0, vibration ×1.4–1.8.

Faults start randomly (`--anomaly-prob` per tick) or on demand via the control topic.

**Fleet** — all machines under management, viewed as one population: the Grafana fleet
dashboard aggregates per-machine metrics, and `make fleet MACHINES=25` scales the
simulated plant (README). In the platform design a fleet spans orgs and sites.

---

## Detection

**Score** — the anomaly score of a reading: mean squared reconstruction error of the
autoencoder in *scaled* feature space; **higher = more anomalous**
(`reconstruction_errors`, `ml/scoring.py`). Legacy IsolationForest bundles keep sklearn's
`decision_function` semantics (negative = anomalous).

**Threshold** — the calibrated alarm level for the model score: the
`(1 − FP_BUDGET)` = 99.5 % quantile of reconstruction errors on **held-out healthy
data** (`ml/train.py`, `FP_BUDGET = 0.005`). Score above threshold ⇒ model-flagged.

**Z-guard** — the hard per-feature limit backstop: a reading is limit-flagged if *any*
feature deviates more than `z_guard` standard deviations (default **6.0**,
`DEFAULT_Z_GUARD` in `ml/scoring.py`) from the training distribution's mean/scale. Exists
because a model can in principle reconstruct (and miss) drift; the guard is the certified
hard limit for single-feature excursions.

**Anomaly** — a reading flagged by the hybrid rule: **model hit OR limit hit**
(`score_sample`, `ml/scoring.py`). Only anomalous readings produce events.

**Reason** — the machine-readable trigger attribution on every anomaly, one of:
- **`model`** — reconstruction error above threshold only;
- **`limit`** — z-guard exceeded only;
- **`model+limit`** — both.

Exact mapping in `score_sample` (`ml/scoring.py`); healthy readings have reason `None`
and never leave the node. Exported as the `reason` label on
`edgesense_anomalies_total`.

**Event** — the JSON message emitted for one anomalous reading:
`{machine_id, ts, score, reason, reading, agent_ts}` — `ts` is the original reading
timestamp, `reading` embeds the full sample, `agent_ts` is when the agent scored it
(`event` struct, `edge-agent/main.go`). Published with QoS 1 to
`edgesense/events/<machine_id>` on the uplink broker. Events are the *only* thing that
goes upstream.

**Episode** — one contiguous fault occurrence: a run of consecutive faulty readings from
a single injected fault, 20–40 ticks when started randomly (`simulator/simulate.py`).
The unit of offline evaluation: an episode counts as detected if *any* of its readings is
flagged (`ml/evaluate.py`).

**Time-to-detect** — latency from the first faulty reading of an episode to the first
flagged reading, reported in seconds and reading count ("detected on reading #1 after
0.00s") by `scripts/demo.py` and as median/p90 per fault by `ml/evaluate.py`. Current
measured value: 1 reading ≈ 0.5 s at 2 Hz for all three fault types
(`docs/EVALUATION.md`).

**False-positive budget** — the accepted rate of healthy readings flagged anomalous, set
by construction: the threshold is calibrated at the 99.5 % healthy quantile, so ≈ 0.5 %
of healthy readings may alarm (`FP_BUDGET`, `ml/train.py`). Measured: 0.43 % in offline
evaluation, 0 across 171 healthy readings in the live demo (README).

**Contamination** — the IsolationForest hyperparameter naming the same concept for the
legacy baseline: the assumed fraction of anomalies in training data, wired to
`FP_BUDGET` (0.005) in `ml/train.py`. Kept only for the `--model iforest` comparison
baseline.

---

## Transport & reliability

**Uplink** — the connection from the edge agent to the remote events broker
(`EDGESENSE_UPLINK_BROKER`; defaults to the local broker, giving single-broker mode).
The link assumed to be flaky (LTE, satellite); its state is exported as
`edgesense_uplink_connected` and in `/healthz` (`edge-agent/main.go`, `metrics.go`).

**QoS 1 / at-least-once** — the MQTT delivery contract used for events: the broker
acknowledges each publish, unacknowledged publishes are retried, so an event arrives *at
least* once (duplicates possible at the protocol level). The agent publishes events with
QoS 1 and a 2 s wait (`publishEvent`, `edge-agent/main.go`). Sensor subscriptions use
QoS 0 — losing a raw reading is acceptable; losing an event is not.

**Store-and-forward** — the agent's offline-event mechanism: publishes are gated on a
*live* uplink connection; on failure the event is appended to a disk-backed FIFO and
delivered later. The gate exists so the disk buffer is the *single owner* of offline
events — otherwise the MQTT library's internal queue would also retry them and produce
duplicates after an outage (comment in `publishEvent`, `edge-agent/main.go`). Buffer
file: `EDGESENSE_BUFFER` (JSON Lines), capacity 10 000 events, **oldest dropped first**
when full, atomically rewritten (temp file + rename), survives agent restarts
(`edge-agent/buffer.go`).

**Buffer depth** — the number of events currently waiting in the store-and-forward
buffer: `EventBuffer.Len()`, exported as the gauge `edgesense_buffer_depth` and the
`buffer_depth` field of `/healthz`. Climbs during an outage, must drain to zero after
replay (`make demo-offline` asserts exactly this).

**Replay** — draining the buffer after the uplink returns: events are published in FIFO
order, stopping at the first failure (the failed event and everything after it stay
buffered) — `DrainTo`, `edge-agent/buffer.go`. Triggered on uplink reconnect and every
30 s (`flushInterval`, `main.go`). Replayed events keep their original reading
timestamps; combined with the connection-gated publish, replay is duplicate-free
("exactly once, with original timestamps" — README, verified by `make demo-offline`).

**Receiver** ***(proposed — roadmap)*** — the cloud-side ingestion service for events:
an MQTT subscriber on `es/+/+/+/events` (and later a CoAP endpoint) that writes events
durably with the org taken from the topic. Does not exist today: the Streamlit dashboard
subscribes directly to the cloud broker (`dashboard/app.py`).

**CON (CoAP)** ***(proposed — roadmap)*** — a *confirmable* message in CoAP, the
UDP-based protocol the README lists as a future uplink for constrained/LTE links. A CON
must be acknowledged by the receiver and is retransmitted otherwise — the CoAP analog of
the QoS 1 at-least-once contract, without per-device connection state.

---

## Platform & identity

*All terms in this group are **proposed** by [`PLATFORM.md`](PLATFORM.md) unless noted;
today both brokers accept anonymous clients with no ACLs (`deploy/mosquitto.conf`).*

**Tenant / Organization** — the isolation boundary of the platform: owns users, sites,
devices, events, and dashboards. Nothing crosses org boundaries except explicit share
grants. Every device is owned by exactly one org at any time.

**Site** — a grouping of machines within an org (plant, hall, vessel). The scope at
which roles are commonly delegated; part of the device alias and topic prefix
(`es/<org>/<site>/<machine>/…`).

**Owner (org owner)** — the top human role of a tenant: manages users and role bindings,
sites, device lifecycle (provision/claim/release/transfer), share grants, and credential
rotation. Full permission matrix in `PLATFORM.md` §3.2.

**Operator** — the day-to-day human role: may inject test faults on machines in scope,
acknowledge/annotate events, and view dashboards. May not provision devices or manage
users.

**Viewer** — read-only human role: dashboards, events, and metrics within scope; no
control actions. Also the effective role granted by an expiring share link.

**Device principal** — the *publish-only* identity a device uses on the uplink: it may
publish under its own topic prefix and subscribe to its own `…/control` topic — nothing
else, enforced by broker ACLs. Phase 1: per-device username/password; phase 2: mTLS
certificate with CN = `device_uid`. Distinct from human and service principals; a
compromised device cannot read other devices' data.

**Provisioning** — enrolling a device into the registry and giving it credentials. mTLS
flow: one-time bootstrap token → device generates its keypair on-device → CSR to the
registry → registry validates the token and signs a short-lived certificate
(`PLATFORM.md` §4.3).

**Claim** — taking ownership of an `UNCLAIMED` or `RELEASED` device using its one-time
claim code (printed/QR on the unit). Requires org-owner or site-admin rights in the
claiming org; atomically sets the owner, site, topic prefix, and broker ACLs.

**Ownership transfer** — moving a device between orgs via *release by the current owner*
→ *claim by the new owner* (e.g. OEM ships a machine, customer claims it). The device's
certificate and keys survive the transfer — only registry authorization and ACLs change.
Event history stays with the org that owned the device when the events were produced.

**Registry** — the one proposed new service: system of record for orgs, sites, devices,
users, role bindings, provisioning tokens, claim codes, certificates, share grants, and
event ack-state. Issues broker credentials/ACLs, signs device CSRs (platform CA), and
feeds Prometheus service discovery. SQLite first, Postgres when needed. The agent talks
to it only at provisioning/renewal, so a registry outage never interrupts detection or
event delivery.

**Broker ACL** — the per-principal topic permission rules enforced by the MQTT broker
(e.g. Mosquitto `acl_file` patterns or the 2.x dynamic-security plugin): a device may
`write es/<its-prefix>/…` and `read es/<its-prefix>/control` only. The mechanism that
makes the device principal publish-only. *Current state: none — `allow_anonymous true`.*

**mTLS (mutual TLS)** — TLS where the *client* also presents a certificate, so both
sides authenticate cryptographically. Phase 3 device identity: the uplink listener
requires client certs, the certificate CN (= `device_uid`) becomes the broker username,
rotation is EST-style renewal of short-lived certs, revocation is registry-driven
disable plus short lifetimes.

---

## Security

*Terms used by [`SECURITY.md`](SECURITY.md). All controls in this group are
**proposed** — today both brokers are anonymous and no link uses TLS
(`deploy/mosquitto.conf`, `edge-agent/uplink.go`); the only current entries are the
threat terms, which name attacks the current state permits.*

**TLS (Transport Layer Security)** — the standard channel encryption + server
authentication protocol; for MQTT it runs on port 8883 (`ssl://` URLs, which the
agent's uplink already routes to paho unchanged — the seam exists, the configuration
doesn't). Server-side TLS is the `SECURITY.md` phase P0 step; **mTLS** (client certs
too) is defined under [Platform & identity](#platform--identity).

**DTLS (Datagram TLS)** — TLS adapted to UDP datagrams (DTLS 1.2, RFC 6347; 1.3,
RFC 9147), the standard way to secure CoAP (`coaps://`, port 5684). PSK mode avoids
certificate weight on constrained links; Connection IDs (RFC 9146) survive NAT
rebinding without re-handshakes. `edge-agent/uplink.go` currently rejects `coaps://`
as "not supported yet" *(current)*; DTLS-PSK is the P5 target. Hop-by-hop: protection
ends at the CoAP receiver.

**OSCORE** — Object Security for Constrained RESTful Environments (RFC 8613): encrypts
each CoAP request/response at the message layer instead of the transport, so protection
is end-to-end across proxies and needs no handshake after link drops — a natural fit
for the agent's store-and-forward uplink. Tracked as a follow-up to DTLS in
`SECURITY.md` §3.2 because mature Go support is lacking.

**ACL (access control list)** — per-principal allow rules; on an MQTT broker,
per-username topic read/write patterns. The P0 quick win is an ACL making
`edgesense/control/fault` writable only by an `ops` user — closing today's
anyone-can-inject-faults hole. Broker mechanics under
[Broker ACL](#platform--identity).

**SBOM (software bill of materials)** — a machine-readable inventory (SPDX/CycloneDX)
of every component in an artifact, generated at build time (e.g. syft) so
vulnerability answers ("do we ship xz 5.6.0?") are lookups, not archaeology. Planned
per container image in CI (P1).

**Secure boot** — firmware verifies the signature of each boot stage (bootloader →
kernel → OS) so only trusted code runs; on Ubuntu Core devices it combines with TPM-
backed full-disk encryption. Relevant only for real fleet hardware — noted in
`SECURITY.md` §5, out of scope for the Docker demo.

**TPM / secure element** — tamper-resistant hardware that generates and holds private
keys as non-exportable objects (TPM 2.0, ATECC608). Target home for the device's mTLS
key (P3): the agent signs via the chip (PKCS#11 / go-tpm), so malware can use the key
while running but never steal it.

**Provisioning token (bootstrap token)** — the one-time, expiring secret that lets a
factory-fresh device request its first certificate: token + CSR in, signed cert out
(`PLATFORM.md` §4.3). Single-use and bound to a `device_uid`, so a leaked token can
enroll at most one impostor, and only until it expires.

**Revocation** — invalidating a device identity before its credential expires:
registry-driven dynsec disable at the broker (immediate refusal, no device contact
needed) plus short cert lifetimes as the backstop, with CRLs deliberately secondary
(`PLATFORM.md` §4.3). Step 1 of decommissioning (`SECURITY.md` §5).

**Data poisoning** — corrupting *training* data so the learned model itself is wrong —
for EdgeSense: feeding abnormal readings labeled as healthy so a retrained autoencoder
learns to accept a developing fault. No field-data training path exists today
*(current)*, so the live risk is nil until retraining lands — which is why
authenticated sensor ingest (P0) is a prerequisite for it (`SECURITY.md` §2.4).

**Evasion** — crafting *inference-time* inputs that a correct model still misses: fault
signatures held under both the reconstruction-error threshold and the 6σ z-guard
(`ml/scoring.py`). Today's open `/score` endpoint is a free tuning oracle for this
(`SECURITY.md` §2.4); a network attacker can add evasive readings but cannot remove
the real sensor stream.

---

## Hardware

*Terms introduced by the hardware chapter ([`HARDWARE.md`](HARDWARE.md)), which replaces
the simulator with real sensors. Today all sensing is simulated (`simulator/simulate.py`);
components below marked **(proposed)** do not exist in the repo.*

**MEMS accelerometer** — a micro-electro-mechanical vibration sensor on a chip
(ADXL355 class): digital output (SPI/I2C), low cost, kHz-range bandwidth. The dev/pilot
choice for producing the `vibration` feature (`HARDWARE.md` §3.1).

**IEPE** — Integrated Electronics Piezo-Electric, the industrial accelerometer standard:
a piezo sensor with built-in charge amplifier powered by a constant-current supply over
its coax cable. Widest bandwidth and the reference-grade option for vibration
(`HARDWARE.md` §3.1).

**RTD / PT100** — Resistance Temperature Detector: a platinum resistor (100 Ω at 0 °C
for PT100) whose resistance varies precisely with temperature; read by a converter such
as the MAX31865. The default source of the `temperature` feature, ring-lug-mounted on a
bearing housing (`HARDWARE.md` §3.2).

**CT clamp (current transformer)** — a split-core transformer clipped *around* one
conductor that outputs a scaled, galvanically isolated image of the AC current — the
source of the `current` feature. Mains-side installation is qualified-electrician work
(`HARDWARE.md` §3.3).

**IO-Link** — a point-to-point industrial sensor protocol (IEC 61131-9): smart sensors
plug into an IO-Link master and report calibrated digital values. IO-Link vibration
transmitters (IFM VVB class) output mm/s RMS directly — the closest physical match to
the reading schema (`HARDWARE.md` §3.4).

**Modbus** — a simple, ubiquitous industrial register-polling protocol (RTU over RS-485,
or TCP). How the adapter reads power meters (SDM class) for `current`, and existing
machine electronics where available (`HARDWARE.md` §3.3–3.4).

**OPC UA** — the machine-to-IT interoperability standard for industrial equipment; PLCs
and VFDs expose telemetry over it. Where a machine already publishes usable signals, the
adapter polls OPC UA instead of adding physical sensors (`HARDWARE.md` §3.4).

**DIN rail** — the standardized 35 mm mounting rail inside electrical cabinets.
"DIN-rail gateway/PSU" = designed to snap into a cabinet next to existing gear — the
form factor for Tier B edge nodes (`HARDWARE.md` §2, §5.2).

**PoE (Power over Ethernet)** — powering a device over its network cable (802.3af/at).
One-cable installs for edge nodes: RPi via PoE+ HAT, some industrial gateways natively
(`HARDWARE.md` §5.1).

**Signal conditioning** — everything between transducer and digital value: filtering,
amplification, anti-aliasing, integration (acceleration → velocity), analog-to-digital
conversion. For vibration it is the step that turns kHz raw samples into the schema's
single mm/s RMS number (`HARDWARE.md` §4.1).

**RMS window** — the feature-extraction step behind the 2 Hz contract: root-mean-square
of a high-rate signal over one 0.5 s reporting tick, published as the reading's value.
`vibration` is RMS of kHz-sampled velocity; `current` is RMS over whole mains cycles.
Matches the simulator's units (mm/s RMS, A) so trained models transfer
(`HARDWARE.md` §4.1–4.2).

**Sensor adapter** ***(proposed)*** — the per-node component that replaces the
simulator: reads sensor drivers (I2C/SPI/UART/Modbus/IO-Link), computes window features,
and publishes schema-conformant JSON readings to `edgesense/sensors/<machine_id>` at
~2 Hz. Fills exactly the simulator's role; the rest of the stack cannot tell the
difference (`HARDWARE.md` §4.3).

**EMI (electromagnetic interference)** — electrical noise coupling into sensor signals,
dominated on plant floors by VFD output cabling. Countered by cable separation, shielded
twisted pair grounded at one end, and preferring digital-at-the-sensor interfaces over
long analog runs (`HARDWARE.md` §5.3).
