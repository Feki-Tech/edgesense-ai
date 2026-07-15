# EdgeSense AI — Hardware chapter: from simulator to real machines

The founding document for the EdgeSense hardware team. Today the entire "plant" is
`simulator/simulate.py` — three virtual machines publishing physics-plausible readings
over MQTT. This chapter defines how to replace that simulator with real sensors on real
industrial machines **without changing a single line of the software stack**.

Definitions of *current* software behavior are grounded in the code (file references
given). Everything describing hardware, parts, and processes is ***(proposed)*** — none
of it exists in the repo today. Prices are rough **2026 estimates** for planning only,
not quotes.

Contents:
[1. Mission & scope](#1-mission--scope) ·
[2. Edge compute tiers](#2-edge-compute-tiers) ·
[3. Sensor selection](#3-sensor-selection-per-signal) ·
[4. Signal conditioning & the 2 Hz contract](#4-signal-conditioning--the-2-hz-contract) ·
[5. Connectivity & power](#5-connectivity--power) ·
[6. Bill of materials](#6-bill-of-materials-two-worked-examples) ·
[7. Software touchpoints](#7-software-touchpoints-for-the-hardware-team) ·
[8. Team charter & milestones](#8-team-charter--milestones)

---

## 1. Mission & scope

**Mission:** put real vibration, temperature and current sensors on real machines and
make them publish the exact same readings the simulator publishes today. Everything
downstream — agent, inference, store-and-forward uplink (MQTT or CoAP), dashboard,
Grafana, demos — keeps working unmodified.

### 1.1 The contract

The hardware/software boundary is **one MQTT message**: a JSON reading published to
`edgesense/sensors/<machine_id>` on the node-local broker, at ~2 Hz (the simulator's
default `--interval 0.5`, `simulator/simulate.py`). The agent's parser
(`reading` struct, `edge-agent/main.go`) defines the schema:

| Field | Type | Unit | Healthy operating point (simulator) | Notes |
|---|---|---|---|---|
| `machine_id` | string | — | `machine-01` … | Optional in the payload: the agent falls back to the last topic segment (`topicMachineID`, `edge-agent/main.go`) |
| `ts` | float | Unix seconds | — | Reading timestamp, set at acquisition. Preserved end-to-end through buffering and replay |
| `vibration` | float | **mm/s RMS** | 0.8 ± 0.15 | Velocity RMS — *not* raw acceleration. See §4 |
| `temperature` | float | **°C** | 45 ± 1.2 | Contact temperature at the measurement point |
| `current` | float | **A** (RMS) | 12 ± 0.6 | One motor phase, or the aggregate the model was trained on |

Rules of engagement:

- **Extra fields are tolerated, not read.** Go's JSON decoding ignores unknown keys —
  the simulator already ships a `fault_injected` field the agent never sees. The
  adapter may add diagnostic fields (e.g. `raw_peak`, `sensor_status`) freely, but
  nothing downstream may *depend* on them without a joint schema change (§8.1).
- **Missing features are not tolerated.** The model's feature vector is exactly
  `["vibration", "temperature", "current"]` (`FEATURES`, `ml/train.py`). An absent key
  decodes as `0.0`, which is itself a >6σ excursion for temperature/current and will
  trip the z-guard — a dead sensor becomes a false "anomaly" instead of an error.
  Sensor-fault handling lives in the adapter (§4.3).
- **Rate is ~2 Hz per machine.** Nothing in the agent hard-codes it, but the measured
  detection latencies ("1 reading ≈ 0.5 s", `docs/EVALUATION.md`), the bandwidth math
  (~50 MB/day/machine, README) and the dashboards all assume it. Treat 2 Hz as the
  contract; renegotiate explicitly if a machine needs different.
- **Topics:** flat `edgesense/sensors/<machine_id>` today; `docs/PLATFORM.md` §4.4
  proposes namespaced `es/<org>/<site>/<machine>/sensors/<sensor>` topics (Phase 1 of
  the platform roadmap). The sensor adapter (§4.3) must take its topic prefix from
  configuration so the migration is a config change, not a firmware change.
- **QoS 0 for readings.** The agent subscribes to sensors at QoS 0
  (`edge-agent/main.go`) — losing a raw reading is acceptable; losing an *event* is
  not. The adapter should publish readings QoS 0 and not build its own retry logic.

### 1.2 In and out of scope

| In scope (hardware team owns) | Out of scope (frozen software contract) |
|---|---|
| Sensor selection, mounting, wiring, conditioning | Agent, inference service, scoring rule |
| The sensor-adapter component (§4.3) *(proposed)* | Event schema `{machine_id, ts, score, reason, reading, agent_ts}` |
| Edge compute selection, enclosures, power | Uplink transports (MQTT QoS 1 / CoAP CON — both shipped) |
| Healthy-baseline data collection & calibration runs (§7.3) | Store-and-forward semantics, metrics names |
| Installation safety & electrical compliance | Dashboard, Grafana, Prometheus, demos |

One deliberate loss: the on-demand fault injection (`edgesense/control/fault`) is a
*simulator* feature — you cannot command a real bearing to fail. Live demos on real
hardware use induced load anomalies on the bench rig instead (§8.2, M0).

---

## 2. Edge compute tiers

The deployment model (matching the snap packaging and `docs/PLATFORM.md` §6) is **one
agent per machine/node**: local broker + agent + inference sidecar on the node, only
events leave. Three hardware tiers cover dev bench to plant floor:

| | Tier A — dev / pilot | Tier B — industrial | Tier C — microcontroller |
|---|---|---|---|
| Example hardware | Raspberry Pi 4/5 (arm64) | DIN-rail fanless gateways: Siemens SIMATIC IOT2050, Advantech UNO-2xxx, OnLogic Karbon 400/800 class | ESP32 / ESP32-S3 class MCU boards |
| CPU / RAM | 4-core Cortex-A72/A76, 4–8 GB | 2–4 core arm64 or x86-64, 2–16 GB | 240 MHz dual-core, ~0.5 MB SRAM |
| Rough unit cost (2026 est.) | $60–100 (+PSU, storage, case) | $500–1,500 | $5–20 |
| Environment | Office/bench; needs a case and clean power on the floor | 24 V DC, fanless, −20…60 °C, DIN-rail, CE/UL, some with UPS/SuperCAP | Anywhere; industrial carriers exist |
| Runs | Full node: mosquitto + agent + inference (Docker or snap) | Same, hardened; Ubuntu Core + snap is the target | **Readings only** — cannot run the stack |
| Role | M0 bench rig, M1 pilot (§8.2) | M2/M3 production nodes | Remote sensor head feeding a Tier A/B gateway |

**Tier C clarification.** The agent is a Linux binary and the inference sidecar is
CPython — neither runs on an MCU. An ESP32 node samples its sensors, computes the
window features (§4), and publishes the schema-conformant JSON reading over Wi-Fi/
Ethernet MQTT to the *gateway's* local broker; the agent on the gateway scores it like
any other machine (the default subscription `edgesense/sensors/#` already fans in any
number of machines, `EDGESENSE_SENSOR_TOPIC`). This turns one Tier B gateway into the
node for a cluster of nearby machines. The **CoAP uplink** (shipped — README "CoAP
uplink", merged PR #4) is *not* for this hop: it covers the gateway→cloud leg over
constrained/LTE links. The sensor-head→gateway hop stays plain MQTT on the local
network. *(A CoAP listener for sensor heads is imaginable but unnecessary today —
propose it only if a plant demands battery-powered wireless heads.)*

### 2.1 Resource footprint (what the node must carry)

Grounded in the shipped artifacts:

| Component | Artifact | Footprint | Source |
|---|---|---|---|
| Edge agent | Static Go binary (`CGO_ENABLED=0`), distroless image | binary ~15 MB, RSS ~10–20 MB, negligible CPU at 2 Hz | `edge-agent/Dockerfile` (`gcr.io/distroless/static-debian12`) |
| Inference sidecar | `python:3.12-slim` + numpy/scikit-learn/FastAPI | image ~400–500 MB, RSS ~100–150 MB; scoring is pure numpy on a 3→16→2→16→3 net — sub-ms per reading | `inference/Dockerfile`, `inference/requirements.txt`, `ml/scoring.py` |
| Local broker | `eclipse-mosquitto:2` | RSS ~5–10 MB | `docker-compose.yml` |
| Event buffer | JSON Lines file, capped 10,000 events | ≤ ~3 MB on disk (~300 B/event) | `edge-agent/buffer.go`, `docs/PLATFORM.md` §6.1 |

Two consequences:

- **Any Tier A/B device is comfortably oversized** — a Pi 4 with 2 GB runs the whole
  node with an order of magnitude of headroom. Sizing is driven by environment
  (temperature, vibration, power quality), not compute.
- **The Python sidecar is the only heavy piece, and it is scheduled for removal:** the
  README roadmap's "Run ONNX inference inside the Go agent (onnxruntime bindings), drop
  the sidecar" item. The export already exists — `make export-onnx` emits a ~2 KiB
  self-contained graph (scaler + weights + threshold baked in, parity-tested in
  `tests/test_onnx.py`). When that lands, a node is a single ~15 MB process plus
  mosquitto, and Tier B minimum specs drop to "anything that boots Linux". Plan Tier C
  gateways assuming the sidecar exists; treat in-agent ONNX as an upside.

---

## 3. Sensor selection per signal

Per-signal options, dev-grade → industrial-grade. Example parts are *classes*, not
endorsements; unit costs are rough 2026 estimates.

### 3.1 Vibration (`vibration`, mm/s RMS)

The hardest channel: the schema wants ISO-style velocity RMS, which no raw
accelerometer outputs directly (see §4).

| Class | Example parts | Interface | Rough cost | Fit |
|---|---|---|---|---|
| MEMS accelerometer (dev/pilot) | ADXL355 (low noise, ±2–8 g), ADXL357; breakout boards | SPI/I2C, 4 kHz ODR | $30–60 breakout | M0/M1 workhorse: low noise floor (~25 µg/√Hz) resolves 0.8 mm/s-class signals; adapter computes RMS (§4) |
| MEMS IMU module with onboard processing | WitMotion WT901 class | UART/I2C, pre-filtered | $40–70 | Quick bench experiments; onboard filtering is opaque — calibrate against a reference before trusting for baselines |
| Industrial MEMS vibration transmitter | IFM VVB001/VVB011 class (IO-Link), Banner QM30VT class (Modbus/1-wire analog) | IO-Link / Modbus / 4–20 mA | $250–450 | **Best schema fit**: computes v-RMS (mm/s, ISO 10816 band) *inside the sensor* — the adapter just polls a number at 2 Hz |
| IEPE piezo accelerometer | PCB 603C01 class, CTC AC102 class | IEPE (constant-current 2-wire coax) → conditioner + ADC, or 4–20 mA vibration transmitter | $250–700 + $150–400 conditioning | Gold standard: widest bandwidth (0.5 Hz–10 kHz+), best for later spectral/bearing-frequency work; most integration effort |

Mounting matters as much as the part: stud-mount or epoxy-mount on the bearing housing,
in the load direction; magnet mounts only for surveys. Keep the mounting resonance well
above the measured band.

### 3.2 Temperature (`temperature`, °C)

| Class | Example parts | Interface | Rough cost | Fit |
|---|---|---|---|---|
| RTD (PT100/PT1000) + converter | PT100 probe/ring-lug + MAX31865 | SPI | probe $10–40, converter breakout $10–20 | **Default choice**: ±0.1–0.5 °C on the bearing housing; ring-lug RTD under a housing bolt is a classic PdM install |
| Thermocouple + converter | Type K + MAX31856 | SPI | $10–30 + $15–25 | Only where >200 °C or fast response is needed; noisier than RTD near 45 °C |
| IR spot sensor (non-contact) | MLX90614 class (dev); industrial IR pyrometer 4–20 mA | I2C / analog | $15–25 dev, $150–500 industrial | Rotating or inaccessible surfaces; watch emissivity drift and dirty lenses |
| IO-Link temperature transmitter | IFM TA/TV class | IO-Link | $100–250 | When an IO-Link master is already on the node (§3.4) |

The overheat fault signature is a +15…30 °C ramp over minutes (`simulator/simulate.py`)
— any option above resolves it easily; choose on mounting and lifecycle, not precision.

### 3.3 Current (`current`, A RMS)

| Class | Example parts | Interface | Rough cost | Fit |
|---|---|---|---|---|
| Split-core CT clamp + ADC | SCT-013 class (0–100 A, voltage output) + ADS1115/MCP3008 ADC | Analog → I2C/SPI | clamp $10–25, ADC $10–15 | Dev/pilot default: clips around one phase conductor, galvanically isolated by design |
| Hall-effect in-line sensor | ACS712/ACS723 class (≤ 30 A) | Analog | $5–15 | Bench rigs and low-voltage DC rigs **only** — it sits in series with the load; never splice it into plant wiring |
| Modbus power meter | Eastron SDM120/SDM630 class + matched CTs | RS-485 Modbus RTU (or TCP) | $30–150 + CTs | Industrial choice: electrician installs meter + CTs in the panel once; adapter polls current (and gets voltage/power/energy for free) |
| Hall/Rogowski clamp transmitter, 4–20 mA | Various | 4–20 mA → ADC | $80–250 | When the panel standard is 4–20 mA |

> **⚠ Safety — mains-side work.** CT clamps go around live conductors; meters and
> anything line-voltage live in the panel. This work is done by a **qualified
> electrician** under the plant's electrical safety rules (lockout/tagout, local code —
> DGUV/VDE, NFPA 70E, or equivalent). The hardware team's self-serve zone is strictly
> SELV: MEMS sensors, RTDs, IO-Link, and the low-voltage side of already-installed CTs.
> Budget electrician time in every pilot (§6.2).

### 3.4 Cross-cutting: fieldbus options

For Tier B installs, prefer sensors behind standard industrial interfaces over raw
analog: **IO-Link** (point-to-point sensor bus; one USB/Ethernet IO-Link master serves
multiple smart sensors), **Modbus RTU/TCP** (meters, drives), and — where machines
already expose telemetry — **OPC UA** from the machine PLC. In all three cases the
sensor adapter (§4.3) is a *poller*, not a signal-processing engine, which is the
cheapest path to schema-conformant readings. Reading electrical data from an existing
VFD/PLC over Modbus/OPC UA can replace a physical CT entirely — ask the plant first.

---

## 4. Signal conditioning & the 2 Hz contract

### 4.1 2 Hz is the *reporting* rate, not the *sampling* rate

The contract's `vibration` value is a **feature**, not a sample. Bearing and gear-mesh
defects live at hundreds of Hz to several kHz; the standard machine-condition band for
velocity RMS (ISO 10816/20816 family) is roughly 10–1000 Hz. Sampling vibration *at*
2 Hz would alias everything of interest into noise.

The pipeline per 0.5 s reporting tick *(proposed)*:

1. Sample acceleration at ≥ 2.56 × band edge — e.g. **3.2–6.4 kHz** for a 1 kHz band.
2. High-pass (~2–10 Hz) to kill DC/mounting drift; anti-alias low-pass at the band edge.
3. Integrate acceleration → velocity (or use a velocity-output sensor).
4. Compute **RMS over the 0.5 s window** (≈ 1,600–3,200 samples) → one mm/s value.
5. Publish that value as `vibration` in the reading.

This is exactly the number industrial vibration transmitters (IFM VVB class, §3.1)
compute internally — which is why they are the best schema fit: steps 1–4 happen inside
the sensor and the adapter only polls. With raw MEMS/IEPE parts, steps 1–4 run in the
adapter, which is entirely feasible on any Tier A/B device (a 3.2 kHz × 0.5 s RMS is
trivial arithmetic) but must be treated as real-time-ish code (no GC pauses dropping
samples — use the sensor's FIFO, e.g. the ADXL355's, and read in blocks).

The window statistic is deliberately **RMS to match the simulator's semantics**
(`vibration: mm/s RMS`, `simulator/simulate.py`) and the trained model. Peak, crest
factor and kurtosis are valuable for bearing diagnostics — the adapter may compute and
attach them as extra (ignored) fields for future use, but the contract value is RMS.

### 4.2 Temperature and current: direct

- **Temperature** moves over seconds-to-minutes: sample directly at 2 Hz (or slower
  with hold). No conditioning beyond the converter chip's filtering.
- **Current** is 50/60 Hz AC: compute RMS over an integer number of mains cycles per
  tick (e.g. 10 cycles = 200 ms at 50 Hz) from a ≥ 1 kHz sampled CT/hall signal — or
  poll a Modbus meter / IO-Link transmitter that already reports RMS amps.

### 4.3 The `sensor-adapter` *(proposed — does not exist in the repo)*

A small per-node component filling exactly the simulator's role: **read hardware,
publish schema-conformant JSON readings to the local broker.** The rest of the stack
cannot tell the difference — that is the definition of done.

```
        REAL MACHINE                                EDGE NODE (Tier A/B)
 ┌────────────────────────┐      ┌──────────────────────────────────────────────────┐
 │ bearing   motor  panel │      │            ┌─────────────────┐                   │
 │  ┌────┐  ┌─────┐ ┌───┐ │      │            │  sensor-adapter │    (proposed)     │
 │  │MEMS│  │RTD  │ │CT │ │      │  drivers   │  sample @ kHz   │                   │
 │  │accl│  │PT100│ │ADC│─┼──────┼──────────► │  condition      │                   │
 │  └──┬─┘  └──┬──┘ └───┘ │ SPI/ │  I2C/SPI/  │  window → RMS   │                   │
 │     │       │          │ I2C/ │  UART/     │  build reading  │                   │
 │     ▼       ▼          │ 485  │  Modbus/   └────────┬────────┘                   │
 │  conditioning          │      │  IO-Link            │ JSON reading, ~2 Hz, QoS 0 │
 │  (HP filter, IEPE      │      │                     ▼                            │
 │   supply, burden R…)   │      │       edgesense/sensors/<machine_id>             │
 └────────────────────────┘      │            ┌─────────────────┐                   │
                                 │            │  local broker   │ (mosquitto)       │
                                 │            └────────┬────────┘                   │
                                 │                     │ subscribe QoS 0            │
                                 │                     ▼                            │
                                 │            ┌─────────────────┐  POST /score      │
                                 │            │   edge-agent    │ ◄───────────────┐ │
                                 │            │  (Go, shipped)  │ ─────────────►  │ │
                                 │            └────────┬────────┘   ┌───────────┐ │ │
                                 │                     │            │ inference │─┘ │
                                 │   anomaly events    │ QoS 1 +    │ (FastAPI) │   │
                                 │   ONLY              │ disk buffer└───────────┘   │
                                 └─────────────────────┼────────────────────────────┘
                                                       │  MQTT (default)
                                                       │  or CoAP/UDP (constrained/LTE)
                                                       ▼
                                              cloud broker / coap-receiver
                                              → dashboard, Grafana
```

Design points *(all proposed)*:

- **One adapter process per machine**, config-driven: machine_id, topic prefix
  (flat today, `es/<org>/<site>/<machine>/…`-ready per `docs/PLATFORM.md` §4.4),
  broker host/port, per-channel driver + calibration (scale/offset), reporting
  interval (default 0.5 s).
- **Same CLI shape as the simulator** (`--broker`, `--port`, `--interval`) so every
  existing runbook transfers.
- **Sensor-fault handling:** on a dead/erroring channel, *stop publishing* that
  machine's readings and raise an adapter-side health metric — never publish zeros
  (a zero is a 6σ excursion and fires the z-guard as a fake anomaly, §1.1). A
  Prometheus endpoint mirroring the agent's pattern (`edgesense_adapter_*`) makes it
  visible in the existing Grafana stack.
- **Language:** Go, mirroring the agent (static binary, trivial arm64 cross-compile,
  snap-friendly) — final call belongs to the implementation PR, not this document.
- **Deployment:** second strictly-confined snap next to `edgesense-agent`, with the
  hardware interfaces from §7.2.

---

## 5. Connectivity & power

### 5.1 Network

Two distinct legs, with different constraints:

| Leg | Carrier | Notes |
|---|---|---|
| Sensor → adapter | SPI/I2C/UART on-board; RS-485 (Modbus), IO-Link, 4–20 mA for cabinet-distance runs | Analog and coax runs short (< 2–10 m); digital fieldbuses handle cabinet-to-machine distances |
| Sensor head (Tier C) → gateway broker | Ethernet or Wi-Fi, MQTT | Plant Wi-Fi is fine for readings (QoS 0, loss-tolerant); wire anything that also hosts an agent |
| Node → cloud (uplink) | **Ethernet** (preferred), Wi-Fi (pilots), **LTE/LTE-M/NB-IoT or satellite** for remote sites | This is exactly what the shipped **CoAP uplink** is for (README "CoAP uplink", PR #4): UDP, 4-byte header, no keepalive, CBOR events ~35 % smaller than JSON, CON retransmit ≈ QoS 1. Set `EDGESENSE_UPLINK_URL=coap://…` — no other change. Store-and-forward covers outages on any carrier (10k-event disk buffer ≈ tolerates days of typical anomaly rates) |

**PoE:** 802.3af/at powers Tier A/B nodes over the Ethernet drop — RPi 5 via the
official PoE+ HAT (~$25), several industrial gateways natively (PoE-PD option). One
cable per node is a meaningful install saving; confirm the switch's PoE budget.

### 5.2 Power

| Item | Choice | Rough cost |
|---|---|---|
| Cabinet supply | 24 V DC DIN-rail PSU (Mean Well MDR/SDR-60 class), sized 2× load | $30–80 |
| Tier B gateway input | Native 9–36 V DC — direct from the 24 V rail | — |
| Tier A (RPi) from 24 V | 24→5 V buck (5 A for a Pi 5) or PoE+ HAT | $10–25 |
| Brownout survival | Gateways with SuperCAP/UPS option, or a small DIN-rail UPS module | $50–150 |

The agent tolerates hard power loss by design (buffered events survive restarts —
`edge-agent/buffer.go`, atomic rewrite), so a UPS is about filesystem health and
avoiding gaps in the healthy baseline, not data loss.

### 5.3 Enclosures, cabling, EMI

- **Enclosure:** inside an existing cabinet on DIN rail (preferred), else a wall-mount
  **IP65/IP66** box (polycarbonate ~$30–80) with cable glands; observe the gateway's
  derated fanless temperature range in sealed boxes.
- **EMI near VFDs** — the plant floor's dominant noise source. Rules:
  - Never run sensor/signal cable in the same tray as VFD output (motor) cables;
    cross at 90° if unavoidable; keep ≥ 30 cm separation for parallel runs.
  - Shielded twisted pair for RTD/analog/RS-485; shield grounded at the cabinet end
    only (avoid ground loops). IEPE runs on its own coax.
  - Prefer digital-at-the-sensor options (IO-Link, Modbus, IEPE transmitters with
    4–20 mA out) over long analog runs — current loops and digital buses shrug off
    what will visibly corrupt a 0–1 V CT signal.
  - Ferrites on DC supply lines entering the enclosure are cheap insurance.
- Label every cable with machine_id + channel; the calibration workflow (§7.3) depends
  on knowing exactly which sensor produced which baseline.

---

## 6. Bill of materials (two worked examples)

> All prices are **rough 2026 estimates** for budgeting — expect ±30 %, plus shipping
> and import duties. Neither BOM includes recurring costs (LTE data plan, cloud broker
> hosting).

### 6.1 BOM A — bench prototype (M0 rig)

Goal: a desk-side rig with a small motor/fan whose load can be perturbed by hand, the
full node stack, real sensors — the simulator replaced end to end.

| # | Item | Example part (class) | Qty | Est. unit | Est. total |
|---|---|---|---|---|---|
| 1 | Edge computer | Raspberry Pi 5, 8 GB | 1 | $80 | $80 |
| 2 | PSU + storage + case | 27 W USB-C PSU, 64 GB A2 microSD (or NVMe HAT), case | 1 | $45 | $45 |
| 3 | Vibration | ADXL355 breakout (SPI) | 1 | $45 | $45 |
| 4 | Temperature | PT100 ring-lug probe + MAX31865 breakout | 1 | $30 | $30 |
| 5 | Current | SCT-013 split-core CT + ADS1115 ADC breakout + burden/divider passives | 1 | $30 | $30 |
| 6 | Demo machine | 230 V fan or small bench motor + clamp-on imbalance weight | 1 | $35 | $35 |
| 7 | Switched mains strip with metering socket (safe CT demo point) | — | 1 | $25 | $25 |
| 8 | Wiring, standoffs, proto HAT, ferrites | — | — | $30 | $30 |
| 9 | Enclosure (optional at bench) | Vented project box | 1 | $20 | $20 |
| | **Total** | | | | **≈ $340** |

Software on the rig: stock `make stack` minus the simulator container, plus the
prototype adapter *(proposed)* — dashboard at :8501 shows the real fan.

### 6.2 BOM B — single-machine industrial pilot (M1)

Goal: one production machine instrumented to plant standards, running for weeks,
feeding the parallel-run evaluation (§8.2, M1). Industrial-transmitter route (steps
1–4 of §4.1 inside the sensors) to minimize custom signal code in the field.

| # | Item | Example part (class) | Qty | Est. unit | Est. total |
|---|---|---|---|---|---|
| 1 | Edge gateway | Siemens SIMATIC IOT2050 / OnLogic Karbon 400 class (arm64, 24 V, DIN) | 1 | $900 | $900 |
| 2 | Vibration transmitter | IFM VVB001-class IO-Link MEMS, M8 stud-mount, outputs v-RMS mm/s | 1 | $350 | $350 |
| 3 | IO-Link master | 4-port, USB or Ethernet | 1 | $400 | $400 |
| 4 | Temperature | PT100 ring-lug on bearing housing + DIN-rail RTD transmitter (or IO-Link TA-class) | 1 | $150 | $150 |
| 5 | Current | Eastron SDM630-class Modbus RTU meter + 3 split-core CTs | 1 | $150 | $150 |
| 6 | 24 V DIN PSU | Mean Well SDR-75-24 class | 1 | $60 | $60 |
| 7 | Enclosure & glands | IP65 polycarbonate, DIN rail kit, glands | 1 | $120 | $120 |
| 8 | Cabling | Shielded sensor cable, RS-485 pair, patch leads, labels | — | $100 | $100 |
| 9 | Network | Ethernet drop or LTE router w/ external antenna (CoAP uplink ready) | 1 | $150 | $150 |
| 10 | **Electrician labor** (meter + CT install, LOTO) | plant/contractor | ~0.5 day | $500 | $500 |
| 11 | Contingency / spares (~15 %) | second accelerometer, fuses, spare PSU | — | — | $400 |
| | **Total** | | | | **≈ $3,300** |

Rule of thumb emerging from BOM B: **≈ $2.5–4 k per machine** for a fully industrial
single-machine node, dominated by the gateway and labor — and amortizable, since one
gateway can serve several adjacent machines (Tier C pattern or multi-drop IO-Link/
Modbus), pushing marginal per-machine cost toward the sensor set (~$650–1,000).

---

## 7. Software touchpoints for the hardware team

The minimum the hardware team must know (and request) from the software side.

### 7.1 arm64

Every Tier A and most Tier B devices are arm64. Current state per artifact:

| Artifact | arm64 status today | Action needed |
|---|---|---|
| Go binaries (agent, coap-receiver) | Trivially cross-compilable — already `CGO_ENABLED=0` static builds (`edge-agent/Dockerfile`) | `GOOS=linux GOARCH=arm64 go build` — no code change |
| Docker images | Single-arch builds; no buildx/multi-arch config in the repo | *(proposed)* add `docker buildx build --platform linux/amd64,linux/arm64` to CI for agent, inference, coap-receiver images (inference is pure-Python + numpy/sklearn — arm64 wheels exist for all of it) |
| Snap | `base: core24` (arm64-capable), but `snap/snapcraft.yaml` declares **no `platforms:` stanza** — snapcraft builds for the build host, and the README's install example is an `_amd64.snap` | *(proposed)* add a `platforms:` stanza for `amd64` + `arm64`, and build via `snapcraft remote-build` or an arm64 runner/Pi |

### 7.2 GPIO/I2C/SPI under snap strict confinement

The agent snap needs nothing new — it is strictly confined with only the `network`
plug and only talks MQTT/HTTP (`snap/snapcraft.yaml`). Hardware access concerns the
**sensor-adapter snap** *(proposed)*:

- Plugs to declare: `i2c`, `spi`, `gpio`, and `serial-port` (RS-485/Modbus RTU,
  IO-Link-over-USB masters may additionally need `raw-usb`); `hardware-observe` for
  device discovery.
- On **Ubuntu Core**, these interfaces' *slots* are provided by the device's **gadget
  snap** (e.g. the Raspberry Pi gadget exposes per-bus `i2c-1`, `spi0`, … slots) —
  connections are made explicitly (`snap connect edgesense-adapter:i2c pi:i2c-1`) or
  auto-connected via store assertion for a branded device. On classic Ubuntu, the
  system provides the slots. Budget for this in bring-up: a missing `snap connect` is
  the classic "works in devmode, fails strict" failure.
- Kernel prerequisites are gadget/device-tree level (enabling I2C/SPI overlays on a
  Pi), owned by whoever builds the device image.

### 7.3 Calibration & retraining workflow

The shipped model is trained on **synthetic healthy data mirroring the simulator's
regime** (`normal_data()`, `ml/train.py`: vib 0.8 ± 0.15 mm/s, temp 45 ± 1.2 °C,
current 12 ± 0.6 A). A real machine has a different operating point — running the
stock model against real sensors will either alarm constantly or never. **Every real
machine needs a healthy-baseline recalibration:**

1. **Collect a healthy baseline.** With the machine verified healthy by maintenance,
   record readings across the full duty cycle — all speeds/loads/products/shifts the
   model should call "normal". Target days, not minutes; the trainer's synthetic
   equivalent uses 20k train + 20k calibration samples ≈ 5.5 h of 2 Hz data as a
   floor. Capture: subscribe to `edgesense/sensors/<machine_id>` and archive JSON
   *(proposed: a small `scripts/record_baseline.py`)*.
2. **Train on it.** The training core is already data-source-agnostic:
   `build_bundle(x_train, x_cal, features, backend)` (`ml/train.py`) takes arbitrary
   arrays — it is exactly how `ml/benchmark_public.py` trains this architecture on the
   real AI4I dataset. *(Proposed: a `ml/train.py --from-csv <file>` flag so the
   hardware team never writes Python.)* Threshold calibration is automatic: the
   99.5 % quantile of reconstruction error on the held-out healthy split
   (`FP_BUDGET = 0.005`).
3. **Thresholds recalibrate themselves — verify them anyway.** Both the model
   threshold and the 6σ z-guard (`DEFAULT_Z_GUARD`, `ml/scoring.py`) derive from the
   training distribution's mean/scale, so recalibration is inherent to retraining.
   Sanity-check the resulting bundle against `docs/EVALUATION.md`'s healthy-FP
   methodology: expected ≈ 0.5 % false-positive rate ⇒ at 2 Hz about one alarm per
   ~2 min/machine *if readings were i.i.d.* — real machines have regime dwell, so
   judge over a full duty cycle.
4. **Deploy** the bundle to the node's inference service (today: bake/copy
   `ml/model/model.joblib`; the README roadmap's "model updates as snap refreshes"
   item is the long-term path) and **re-run the parallel comparison** (§8.2 M1).
5. **Re-baseline triggers:** sensor replaced or remounted, machine overhauled,
   process/recipe change, seasonal ambient shift. A swapped accelerometer with a
   different mounting is a new sensor — treat its old baseline as void.

Per-machine models are the default assumption (operating points differ); revisit
fleet-level models only with M2 data in hand.

### 7.4 What the hardware team may *not* change

The freeze list, for the avoidance of doubt: reading schema and units (§1.1), topic
shape (until PLATFORM Phase 1 lands for everyone at once), event schema, agent env-var
contract (README "Agent configuration"), QoS semantics, metrics names. Changes here go
through the joint process in §8.1.

---

## 8. Team charter & milestones

### 8.1 Responsibilities & interface control

| | Hardware team | Software team |
|---|---|---|
| Owns | Sensors → schema-conformant readings on the local broker: selection, mounting, conditioning, the sensor adapter, node hardware, power/enclosures/EMI, installation & safety compliance, healthy-baseline collection | Everything from the broker subscription down: agent, inference, scoring, uplinks, buffer, dashboards, metrics, CI, packaging of the shipped components |
| Shared | Baseline → retrain → deploy loop (§7.3): hardware runs collection & install; software runs training & bundle deployment. Milestone reviews. | |
| Interface | **The MQTT reading contract (§1.1) is frozen.** Any change — field, unit, rate, topic — requires a PR touching this doc + `docs/GLOSSARY.md`, sign-off from both teams, and a compatibility statement for the trained models. | |

Escalation: anything ambiguous about "is this hardware or software" defaults to
*whoever owns the side of the broker it runs on*; the adapter is hardware-team-owned
even though it is software.

### 8.2 Milestones

| Milestone | Deliverable | Definition of done | Depends on |
|---|---|---|---|
| **M0 — bench rig** (BOM A) | Fan/motor rig with synthetic load faults (imbalance weight, airflow blocking, added friction), full node stack, prototype adapter | Dashboard + Grafana show the real rig; induced faults produce events with sensible `reason` attribution; a screen-recorded equivalent of `make demo` (manual faults, since `edgesense/control/fault` is simulator-only, §1.2) | Adapter prototype; arm64 images (§7.1) |
| **M1 — single-machine pilot** (BOM B) | One production machine instrumented; **parallel run**: real adapter and simulator publish side-by-side under distinct `machine_id`s to the same stack, proving zero software delta; healthy baseline collected; model retrained per §7.3 | ≥ 2 weeks unattended; uplink outage recovery observed in the wild (buffer fills & drains — the `make demo-offline` behavior on real infrastructure); pilot report: FP rate over duty cycle vs the 0.5 % budget, any missed events vs maintenance log | M0; electrician; plant access |
| **M2 — 3-machine mini-fleet** | Three machines (≥ 2 machine types) on per-machine models; evaluation report following `docs/EVALUATION.md` methodology (episodes detected, median/p90 time-to-detect, reading recall, FP rate) — real fault episodes from maintenance logs where available, induced faults on the M0 rig for controlled recall numbers | Report reviewed by both teams; go/no-go + per-machine cost actuals vs §6 estimates; decision on per-machine vs fleet models | M1 report |
| **M3 — fleet rollout** | Rollout wave aligned with `docs/PLATFORM.md` §7 phases: namespaced topics + broker auth (Phase 1) and registry/per-device credentials (Phase 2) land *with* the physical rollout, so hardware is never re-touched for the identity migration | Per-site runbook; installation time per machine below target; fleet visible in Grafana per `PLATFORM.md` §6 scaling analysis | M2 go; PLATFORM Phase 1–2 |

Cadence *(proposed)*: weekly hardware/software sync during M0–M1; milestone reviews
written, with this document updated as the single source of truth.

---

*Terminology used here (IEPE, IO-Link, RMS window, sensor adapter, …) is defined in
[`docs/GLOSSARY.md`](GLOSSARY.md#hardware); the platform-scale context lives in
[`docs/PLATFORM.md`](PLATFORM.md).*
