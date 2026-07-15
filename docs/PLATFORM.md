# EdgeSense AI — Platform Design

**From single-plant demo to multi-user platform**

| | |
|---|---|
| Status | **Proposal** — phases marked *shipped* are implemented (phase 1: PR #7); everything else is design intent |
| Scope | Tenancy & ownership, users/roles/sharing, device identity, registry service, scalability, roadmap |
| Non-goals | Changing the detection pipeline (model, scoring, store-and-forward) — it already works and is kept as-is |
| Companion | [`GLOSSARY.md`](GLOSSARY.md) — definitions of every term used here and in the repo |

---

## 1. Where we are today (current state)

Everything in this section is grounded in the code on `main`; file references point at the
source of truth.

The repo is a **single-tenant, single-plant demo** with a deliberately clean edge design:

```
┌───────────┐ sensors  ┌────────────┐ /score  ┌────────────┐
│ simulator │ ───────► │ edge-agent │ ──────► │ inference  │
│ 3 machines│   MQTT   │    (Go)    │ ◄────── │ (FastAPI)  │
└───────────┘  (local) └─────┬──────┘   HTTP  └────────────┘
                             │ events only · QoS 1 · store-and-forward
                             ▼
                      ┌──────────────┐        ┌────────────┐
                      │ cloud broker │ ─────► │ dashboard  │
                      └──────────────┘        └────────────┘
   agent /metrics ──► Prometheus ──► Grafana
```

Current facts that shape this design:

- **Topics** are flat and unauthenticated *by default*: `edgesense/sensors/<machine_id>`,
  `edgesense/events/<machine_id>`, and a single global `edgesense/control/fault`
  (`edge-agent/main.go`, `simulator/simulate.py`). Any client may publish or subscribe to
  anything: both brokers run with `allow_anonymous true` and no ACLs
  (`deploy/mosquitto.conf`). *Since phase 1 shipped (PR #7), the namespaced layout and an
  authenticated uplink broker exist as opt-ins (`EDGESENSE_ORG`/`EDGESENSE_SITE`,
  `make stack-secure`) — see §4.2, §4.4.*
- **`machine_id` is free-form.** It is whatever the payload says, falling back to the last
  topic segment (`topicMachineID` in `main.go`). Nothing prevents two machines from
  claiming the same ID, or one machine from publishing as another.
- **Client identities are fixed strings** (`edgesense-agent`, `edgesense-agent-uplink`,
  `edgesense-simulator`, `edgesense-dashboard-*`) — labels, not identities.
- **Only events go upstream.** Raw telemetry (~2 Hz per machine, ~50 MB/day/machine per the
  README) stays on the local broker; anomaly events are published with QoS 1 and buffered
  on disk (JSONL FIFO, capped at 10 000, oldest dropped first) when the uplink is down
  (`edge-agent/buffer.go`).
- **Grafana runs anonymous with org role `Admin`**, Streamlit has no login at all
  (`docker-compose.yml`). Prometheus scrapes one statically configured agent target every
  5 s (`deploy/prometheus.yml`).
- **The control topic is a demo device.** `edgesense/control/fault` lets *any* MQTT client
  inject faults into *any* simulated machine — fine for a demo, unacceptable for a platform.

**The one-sentence takeaway:** the data-plane architecture (edge scoring, events-only
uplink, store-and-forward) already scales; what is missing for a multi-user platform is
*identity, authorization, and tenancy metadata*. That is what this document adds — as five
independently shippable phases (§7) that keep today's single-tenant mode as the default.

---

## 2. Tenancy & ownership model *(proposal)*

### 2.1 Hierarchy

Four levels, each owned by the level above:

```
Organization (tenant)               acme-pumps
│  billing, users, policies        │
├── Site                           ├── lyon-plant
│   │  physical/logical location   │   ├── Machine  pump-07
│   ├── Machine (≡ Device)         │   │   ├── Sensor vibration   (mm/s RMS)
│   │   │  monitored asset +       │   │   ├── Sensor temperature (°C)
│   │   │  edge node identity      │   │   └── Sensor current     (A)
│   │   └── Sensor                 │   └── Machine  press-02
│   │        one measured channel  └── nantes-plant
│   │                                  └── Machine  compressor-01
```

- **Organization (tenant)** — the isolation boundary. Users, devices, events, dashboards,
  and broker permissions never cross it except through explicit sharing (§2.3).
- **Site** — a grouping of machines (plant, hall, vessel, field cluster). Unit of
  delegation: roles can be granted per site (§3).
- **Machine / Device** — in this codebase the monitored asset and the compute endpoint
  running the agent are 1:1 (the agent is packaged as one snap daemon per node,
  `snap/snapcraft.yaml`), so the platform treats *Machine* and *Device* as one enrollable
  object. Today's `machine_id` becomes its human alias.
- **Sensor** — one measured channel. Currently exactly three per machine (`vibration`,
  `temperature`, `current` — `ml/train.py` `FEATURES`), modeled explicitly so future
  machines can differ.

**Invariant: every device is owned by exactly one organization at any time.** Ownership is
a registry fact (§5), not a certificate fact (§4.3) — deliberately, so transferring a
device never requires re-provisioning it.

### 2.2 Ownership transfer (claim / release)

Motivating flow: *an OEM builds and provisions the machine, ships it, and the customer
claims it.*

```
 OEM org                         Registry                      Customer org
 ───────                        ─────────                      ────────────
 provision device ────────────► device created, owner=OEM
 ship machine + claim code
 release device ──────────────► state: RELEASED (or factory-new:
                                 UNCLAIMED — never had an owner)
                                                    claim(code) ◄─ admin enters
                                owner := customer      claim code
                                site  := chosen by customer
                                broker ACLs rewritten to
                                es/customer/<site>/<machine>/…
                                device told its new prefix on
                                next (re)connect
```

Rules:

1. **Claim** requires a one-time claim code (issued at provisioning, printed/QR on the
   unit) *and* an `org owner`/`site admin` role in the claiming org.
2. **Release** puts the device in `RELEASED` state: it keeps its identity and keys but has
   no owner; it can buffer events (store-and-forward already handles a dead uplink) but
   nothing is routed until it is claimed.
3. Transfer is **atomic in the registry**: owner change + ACL rewrite + topic-prefix
   reassignment commit together.
4. **Event history does not transfer.** Events produced under the OEM's ownership stay in
   the OEM's tenant; the customer's history starts at claim time. (Data ownership follows
   org ownership at the time of production.)
5. The device's **certificate/credentials survive transfer** — only authorization changes
   (§4.3 explains why the cert encodes identity, not ownership).

### 2.3 Cross-org read-only sharing

Motivating flow: *the OEM keeps monitoring the machines it sold, under a support
contract.*

A **share grant** is a registry object:

```
ShareGrant { subject: device | site,  grantor_org,  grantee_org,
             rights: READ,  expires_at,  revocable_by: grantor }
```

- Grantee users see the shared devices in their inventory, marked `shared (read-only)`,
  and can view events, scores, and dashboards for them — never ack events, never inject
  test faults, never reconfigure.
- Enforcement lives in the **registry/API layer and dashboards** (row filters on
  `org` + share grants), *not* in extra broker subscriptions: devices publish only to
  their own org's prefix, and cloud-side consumers read through the API. If a grantee
  insists on raw MQTT, the registry can additionally emit a read-only ACL binding for a
  grantee service principal on the shared subtree — an opt-in, not the default.
- Grants expire (`expires_at` mirrors the support contract) and are revocable at any time
  by the grantor.

---

## 3. Users, roles & sharing (RBAC) *(proposal)*

### 3.1 Principals

| Principal | What it is | Authenticates with |
|---|---|---|
| **Org owner** | Human; administers the tenant | OIDC (SSO) |
| **Site admin / maintainer** | Human; runs one or more sites | OIDC |
| **Operator** | Human; works the machines day-to-day | OIDC |
| **Viewer** | Human; read-only stakeholder | OIDC or share link (§3.3) |
| **Device principal** | One per device; *publish-only* identity the agent uses on the uplink | Phase 1: username/password · Phase 2: mTLS cert (§4) |
| **Service principals** | `inference`, `receiver`, `dashboard-backend`, `prometheus` — platform services | OAuth2 client credentials → short-lived JWT |

Roles are granted as **bindings with a scope**: `(principal, role, scope)` where scope is
an org, a site, or a single machine. A site admin of `lyon-plant` has no rights in
`nantes-plant`.

### 3.2 Role × permission matrix

| Permission | Org owner | Site admin | Operator | Viewer | Device | Service |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Manage users & role bindings (in scope) | ✅ | ✅ site-scoped | — | — | — | — |
| Create/rename sites | ✅ | — | — | — | — | — |
| Provision / claim / release devices | ✅ | ✅ | — | — | — | — |
| Transfer device ownership, create share grants | ✅ | — | — | — | — | — |
| Rotate/revoke device credentials | ✅ | ✅ | — | — | — | — |
| Inject test faults (`…/control`) | ✅ | ✅ | ✅ | — | — | — |
| Acknowledge / annotate events | ✅ | ✅ | ✅ | — | — | — |
| View dashboards, events, metrics (in scope) | ✅ | ✅ | ✅ | ✅ | — | per-service |
| Create expiring share links | ✅ | ✅ | — | — | — | — |
| Publish sensor readings & events (own prefix only) | — | — | — | — | ✅ | — |
| Subscribe to own `…/control` topic | — | — | — | — | ✅ | — |
| Consume events fleet-wide (ingest) | — | — | — | — | — | ✅ receiver |
| Scrape/read metrics | — | — | — | — | — | ✅ prometheus |

Notes:

- The **device principal is publish-only by construction** (broker ACL, §4.4): it cannot
  read other devices' data even if the box is compromised.
- *Inject test faults* is operator-and-up and becomes **per-machine** (the proposed
  `…/<machine>/control` replaces today's global `edgesense/control/fault`, which any
  anonymous client can use).
- *Ack events* is new platform behavior — today events are fire-and-forget onto the cloud
  broker; the registry adds event state (`open → acked → closed`) for maintenance
  workflows.
- The `inference` sidecar today is node-local (`http://inference:8800/score`,
  `docker-compose.yml`) and can stay identity-free on the node; it appears here for the
  variant where scoring is offered as a shared cloud service.

### 3.3 Dashboard sharing

Two mechanisms, both scope-bound:

1. **Role grants per scope** — the normal path. "Give `viewer` on `site:lyon-plant` to
   `jane@customer.com`."
2. **Expiring read-only share links** — for people without accounts (an auditor, the
   OEM's field tech). A share link is a signed token
   `{scope, rights: READ, exp, link_id}`; opening it renders the dashboard for that scope
   only. Links have a default TTL (e.g. 7 days), are listed and revocable in the registry
   (`link_id` is the revocation handle), and never grant ack/control rights.

### 3.4 Grafana mapping

Today's Grafana is anonymous-`Admin` with one provisioned fleet dashboard
(`deploy/grafana/…`) — fine for the demo, the first thing to go in phase 4.

Two workable multi-tenant mappings:

| | **Org-per-tenant** | **Folder-per-tenant** (recommended) |
|---|---|---|
| Isolation | Hard (Grafana orgs share nothing) | Folder + datasource permissions |
| Datasources | One Prometheus datasource per org | One shared Prometheus, per-tenant enforcement |
| Tenant filtering | Not needed (separate datasources) | **Required**: inject `org` label via a PromQL-rewriting proxy (e.g. `prom-label-proxy`) in front of the datasource, since OSS Grafana cannot enforce label filters itself |
| Cross-tenant ops view | Painful (org switching) | Easy (platform-ops folder) |
| Provisioning | Org + datasource + dashboards per tenant via API | One folder + team per tenant via API |
| When to choose | Contractual hard isolation | Default |

Either way: Grafana auth moves to OIDC against the registry, `GF_AUTH_ANONYMOUS_ENABLED`
goes to `false`, and the agent's metrics gain an `org` (and `site`) label — today the
label set is `{machine, reason}` only (`edge-agent/metrics.go`), which cannot be filtered
per tenant.

---

## 4. Device identity *(proposal)*

### 4.1 Stable device ID

Two-part identity:

- **`device_uid`** — UUIDv4, immutable for the device's life, survives ownership
  transfer. Primary key everywhere: events, metrics, certificates.
- **Human alias** — the `org/site/machine` triple (e.g. `acme-pumps/lyon-plant/pump-07`),
  unique within the org, mutable (rename, move between sites, transfer between orgs).

Today's free-form `machine_id` maps onto the alias's last segment; event payloads gain a
`device_uid` field while keeping `machine_id` for compatibility.

### 4.2 Phase 1 — per-device username/password + broker ACLs *(shipped — PR #7)*

Cheapest real step up from `allow_anonymous true`:

- The registry issues each device a broker **username = `<org>/<site>/<machine>`** and a
  random password (shipped: `EDGESENSE_UPLINK_USERNAME/_PASSWORD` and
  `EDGESENSE_BROKER_USERNAME/_PASSWORD` next to the existing broker env vars).
- Mosquitto gets a `password_file` + `acl_file` generated registry-style (phase 2 swaps
  the generator for the **Mosquitto 2.x dynamic-security plugin** driven over its
  `$CONTROL` topics — no reload dance). The original idea — one `pattern` rule for the
  whole fleet, exploiting username = topic triple:

  ```
  pattern write     es/%u/sensors/#      # DOES NOT WORK — kept for the record
  pattern write     es/%u/events
  pattern read      es/%u/control
  ```

  **was disproven during implementation**: Mosquitto 2.x `pattern` ACLs fail when `%u`
  contains slashes — a device publishing under its *own* prefix is denied. Shipped
  mechanism: **generated per-device `user` blocks** (`scripts/gen_broker_auth.py`,
  `deploy/secure/acl` — demo credentials only), verified live by `make check-acl`
  (own-prefix publish delivered; foreign-org and sibling-machine publishes denied).
  A gateway that scores several machines over one connection (the compose agent) gets a
  scoped *site-gateway* credential (`write es/<org>/<site>/+/events` — the "ops tooling,
  scoped" row of §4.4).
- TLS on the uplink listener (server-side cert only in this phase) so passwords never
  cross the WAN in clear. *(Still open — config stub in `deploy/secure/mosquitto.conf`.)*

Limitations (accepted for phase 1): passwords are bearer secrets on the device's disk;
rotation requires touching the device; no cryptographic binding between the connection
and the device.

### 4.3 Phase 2 — X.509 client certificates, mutual TLS

- Uplink listener requires client certs (`require_certificate true`,
  `use_identity_as_username true`), so the **certificate CN becomes the broker username**.
- **CN = `device_uid`** — identity only. Org/site/machine deliberately stay *out* of the
  cert so ownership transfer (§2.2) is a registry update, never a re-issuance. The
  broker's ACL rules are keyed off the registry's `device_uid → topic prefix` mapping
  (dynsec role per device, updated on transfer).
- **Provisioning flow:**

  ```
  installer            device                      registry (CA)
  ─────────            ──────                      ─────────────
  fetch one-time
  bootstrap token ───► stored at first boot
                       generate keypair on-device
                       (private key never leaves)
                       CSR(CN=device_uid) + token ─► validate token (single-use,
                                                     expiring, bound to device_uid)
                       cert (e.g. 90-day) ◄───────── sign with platform CA
                       connect uplink with mTLS
  ```

- **Rotation:** short-lived certs renewed over the existing mTLS channel (EST-style
  re-enroll) well before expiry; the agent already survives uplink loss, so a botched
  renewal degrades to store-and-forward, not data loss.
- **Revocation:** primary mechanism is the registry disabling the device in dynsec
  (immediate disconnect + auth refusal) plus short cert lifetimes; CRL distribution to
  brokers is the backstop, not the front line.

### 4.4 Topic namespace redesign

Today (flat, trust-everyone):

```
edgesense/sensors/<machine_id>      raw readings (local broker)
edgesense/events/<machine_id>      anomaly events (uplink broker)
edgesense/control/fault            global demo fault injection — any client, any machine
```

Proposed (tenant-prefixed, ACL-enforceable):

```
es/<org>/<site>/<machine>/sensors/<sensor>    raw readings   — node-local only
es/<org>/<site>/<machine>/events              anomaly events — uplink
es/<org>/<site>/<machine>/control             per-machine commands (fault injection,
                                              config push, "your prefix changed")
```

ACL matrix on the uplink broker:

| Principal | `…/sensors/#` | `…/events` | `…/control` |
|---|---|---|---|
| Device (own prefix only) | publish¹ | publish | subscribe |
| Device (any other prefix) | ✗ | ✗ | ✗ |
| Receiver service | — | subscribe `es/+/+/+/events` | — |
| Ops tooling (per role) | — | subscribe (scoped) | publish (scoped) |
| Everyone else | ✗ | ✗ | ✗ |

¹ Sensors normally never reach the uplink broker (that is the point of the design); the
rule exists so a *single* ACL scheme also works on shared site-level brokers.

Consequences for existing code (the honest list — this is the entire phase-1 code
surface):

- `EDGESENSE_SENSOR_TOPIC` is already configurable (`edgesense/sensors/#` default), so
  subscription filters just change value.
- `topicMachineID` in `main.go` takes the **last** topic segment; under the new layout
  the machine is the 4th segment, so the fallback parser needs updating (payloads carry
  `machine_id`, so the fallback rarely fires — but it must not silently mis-parse).
- The event publish topic format string (`edgesense/events/%s`) becomes prefix-aware.
- MQTT client options gain credentials/TLS config.
- Simulator/dashboard/demos follow the same substitution. Default org/site
  (`default/default`) keeps the single-tenant quickstart working unchanged (§7).

---

## 5. Registry service concept *(proposal — this doc proposes, it does not implement)*

One small new service — the only new component in the whole design:

```
                    ┌───────────────────────────────┐
   humans (OIDC) ──►│           REGISTRY            │
   devices (CSR) ──►│  orgs · sites · devices ·     │──► Mosquitto dynsec
   services (JWT)──►│  users · role bindings ·      │    (ACLs, credentials)
                    │  provisioning & claim codes · │──► CA: sign device certs
                    │  share grants · event state   │──► Prometheus file_sd
                    │  SQLite → Postgres            │    (scrape targets)
                    └───────────────────────────────┘
```

- **Data model:** `orgs`, `sites`, `devices` (uid, alias, owner, state:
  `UNCLAIMED|ACTIVE|RELEASED|REVOKED`), `sensors`, `users`, `role_bindings
  (principal, role, scope)`, `provisioning_tokens`, `claim_codes`, `certificates`,
  `share_grants`, `share_links`, `events` (ack state only — the payload stream stays on
  MQTT/whatever sink the deployment chooses).
- **Storage: SQLite first, Postgres when needed.** The write rate is administrative
  (provisioning, claims, role edits) — a few writes per minute even for a large fleet —
  so SQLite honestly carries phases 2–3; the schema is kept portable so the swap is a
  connection string, not a migration project.
- **API:** small REST/JSON service; FastAPI keeps it in the repo's existing stack next to
  `inference/server.py`.
- **How the existing pieces authenticate against it:**
  - **edge-agent** — never talks to the registry at runtime. It touches it exactly twice:
    at provisioning (bootstrap token → CSR → cert, §4.3) and at renewal. Day-to-day it
    only holds broker credentials. A registry outage therefore cannot interrupt
    detection or event delivery — store-and-forward semantics are preserved end-to-end.
  - **inference sidecar** — node-local (same box/pod as the agent), stays identity-free;
    only a hypothetical shared scoring service would get a service principal.
  - **receiver** *(roadmap component — see below)* — service principal, client-credentials
    JWT; subscribes `es/+/+/+/events` and writes to the event store with the `org` taken
    from the topic, never from the payload.
  - **dashboard backend** — service principal for data access; enforces the *user's*
    scopes (role bindings + share grants fetched from the registry) on every query.
  - **Prometheus** — consumes a registry-generated `file_sd` target list instead of
    today's static `agent:8890` (`deploy/prometheus.yml`).

*"Receiver" today:* a first receiver **exists now** — `coap-receiver/` accepts the
agent's confirmable CoAP POSTs on `/events` (CBOR, JSON fallback) and republishes them
to the cloud MQTT broker on the *legacy* `edgesense/events/<machine_id>` topics; the
Streamlit dashboard still subscribes straight to the broker (`dashboard/app.py`). The
registry-integrated receiver described here extends that bridge: durable storage, a
service-principal identity, and org taken from the namespaced topic/credential.

---

## 6. Scalability: 100s–1000s of devices *(analysis)*

Assumed deployment model (matches the snap packaging, not the all-in-one compose demo):
**one agent per machine/node**, local broker + inference sidecar on-node (or in-agent ONNX
later, per the README roadmap), one shared uplink broker + one Prometheus per region/fleet.

### 6.1 The numbers

Grounded in the repo's own characteristics: 2 Hz × 3 sensors per machine, one ~150 B JSON
reading per tick, ~300 B events, 0.5 % calibrated false-positive budget
(`FP_BUDGET = 0.005`, `ml/train.py`; measured 0.43 % offline, 0 in the live demo).

| Quantity | Per machine | 500 machines | 1000 machines | Ceiling / comment |
|---|---|---|---|---|
| Raw readings | 2 msg/s, ~50 MB/day | — | — | **Never leaves the node** — by design the dominant volume is local |
| Uplink events, healthy day (observed) | ≈ 0 | ≈ 0 | ≈ 0 | README: "typically zero upstream bytes on a healthy day" |
| Uplink events, worst case at FP budget | ≤ 0.5 % × 172 800 = **864 ev/day ≈ 0.26 MB/day** | ≤ 5 ev/s | ≤ 10 ev/s | Any broker shrugs at 10 msg/s |
| Fault episode burst | 20–40 events over 10–20 s | — | — | Simulator episode length; trivially absorbed |
| Uplink TCP connections | 1 | 500 | 1000 | Mosquitto: 10k+ concurrent conns is routine (fd-limited) |
| Prometheus series (edgesense_*) | ~24¹ | ~12 k | ~24 k | |
| Prometheus series (incl. Go runtime) | ~75 | ~38 k | ~75 k | Single Prometheus is comfortable to ~1M+ series |
| Prometheus samples @ 5 s scrape | ~15/s | ~7.5 k/s | ~15 k/s | Comfort zone ~100k+ samples/s |

¹ From `edge-agent/metrics.go` with one machine per agent: `readings_scored{machine}` 1 +
`anomalies{machine,reason}` ≤ 3 (`reason ∈ model|limit|model+limit`) + 5 scalar
counters/gauges + latency histogram 12 buckets + `+Inf` + `_sum` + `_count` = 15 →
**≈ 24 series**. All labels are bounded — no cardinality traps in the current metric set.
Adding `org`/`site` labels (§3.4) multiplies nothing (they are constant per agent).

### 6.2 MQTT brokers

Mosquitto (current) is single-process but handles **10k+ concurrent connections** with
tuned fd limits, and the message rate here is trivial (§6.1). It carries this design to
low tens of thousands of devices. Consider **EMQX or VerneMQ** not for throughput but
when you need: clustering/HA for the uplink broker (Mosquitto has none natively),
built-in multi-tenancy and per-tenant rate limiting, or ≥ 50k–100k connections. The topic
namespace and ACL model (§4.4) are deliberately portable across all three.

### 6.3 Prometheus

Capacity is a non-issue (table above). What actually breaks first is **discovery and
reachability**: `static_configs: [agent:8890]` doesn't enumerate a fleet, and Prometheus
*pulls* — edge nodes behind NAT/LTE cannot be scraped. Fix: registry-generated `file_sd`
where reachable; where not, a per-site collector (e.g. Prometheus agent mode)
`remote_write`-ing to the central server. Either preserves the existing metric names,
dashboards, and the `make demo-offline` assertions.

### 6.4 Dashboards

Streamlit (current) subscribes to `edgesense/sensors/#` *and* `edgesense/events/#` in one
Python process (`dashboard/app.py`): at 500 machines that is 1 000 raw msg/s into one
collector, replotted per refresh — and Streamlit has no auth, no RBAC, no tenancy.
Verdict: **keep Streamlit as the single-site/commissioning view; go Grafana-first for
fleet operations** (the metrics already exist and the dashboard is already provisioned),
and introduce the web app + OIDC (dashboard backend, §5) when multi-tenant event
workflows (ack, share links) land in phase 4–5. Raw-signal browsing at fleet scale should
read from an event/telemetry store fed by the receiver, not from a wildcard MQTT
subscription.

### 6.5 CoAP receiver fan-in *(shipped since)*

The CoAP uplink has landed since this document was first written: the agent can POST
events as confirmable (**CON**) CoAP messages (`edge-agent/coap.go`, CBOR-encoded), and
`coap-receiver/` bridges them onto the cloud broker, acknowledging **only after** the
broker's QoS 1 PUBACK — so store-and-forward holds end-to-end (a broker outage behind
the receiver returns 5.03 and the agent keeps buffering). The sizing argument stands:
CON over UDP gives the same at-least-once contract as QoS 1 with *no persistent
connection state*, so receiver fan-in is bounded by event rate, not connection count.
At the worst-case ≤ 10 ev/s for 1 000 machines (§6.1), a single receiver instance is
bored; the remaining design point for the platform is **idempotent ingest** (dedupe key
`device_uid + reading ts`), mirroring the exactly-once-after-replay semantics the disk
buffer already provides on MQTT (`buffer.go` drains FIFO, publish gated on live
connection). The receiver currently speaks the legacy topic layout only — adopting the
namespaced layout and a service-principal identity is phase-2 scope.

### 6.6 Verdict

**Feasible without architectural change.** The events-only uplink means 1 000 machines
generate less upstream traffic than one machine would generate raw. Every hot path
(broker msg/s, Prometheus series, receiver ingest) has ≥ 10× headroom at 1 000 devices on
single instances. **The work is identity, authorization, and tenancy metadata — §§2–5 —
not throughput.**

---

## 7. Phased roadmap *(proposal)*

Each phase ships independently, is useful on its own, and **single-tenant demo mode
remains the default** throughout: `make stack` keeps working with an implicit
`default/default` org/site and anonymous local brokers; everything below is opt-in
configuration.

| Phase | Delivers | Touches | Depends on | Done when |
|---|---|---|---|---|
| **1. Topic namespacing + broker auth/ACLs** — ✅ **shipped** (PR #7) | `es/<org>/<site>/<machine>/…` topics; uplink broker requires username/password; generated per-device ACLs (§4.2, §4.4 — pattern ACLs disproven, see §4.2); global fault topic → per-machine `…/control` | agent (topic strings, `topicMachineID`, MQTT creds), simulator, dashboard, demos, mosquitto conf | — | ✅ `make check-acl` proves a device credential publishes only under its own prefix; demos pass in both layouts (uplink TLS still open) |
| **2. Registry + provisioning + per-device creds** | Registry service (§5): orgs/sites/devices/users/roles, provisioning tokens, claim codes; generates broker credentials & ACLs (dynsec); Prometheus `file_sd` | new service; deploy config | 1 | Enrolling a device end-to-end (token → creds → publishing) without hand-editing broker files |
| **3. mTLS device identity** | Platform CA in the registry; CSR signing, 90-day certs, EST-style renewal, revocation via dynsec disable (§4.3); uplink listener `require_certificate` | registry, agent TLS config, broker listener | 2 | Password auth off on the uplink listener; transfer/revoke take effect without touching the device |
| **4. Multi-tenant dashboards** | Grafana → OIDC, folder-per-tenant + label-enforcing proxy (§3.4); `org`/`site` metric labels; dashboard backend + event ack | grafana provisioning, metrics.go labels, dashboard backend service | 2 | Two orgs on one stack cannot see each other's data; anonymous Admin is gone |
| **5. Ownership transfer & sharing UX** | Claim/release flows (§2.2), cross-org share grants (§2.3), expiring share links (§3.3) | registry + dashboard backend | 2–4 | OEM→customer transfer works with zero device-side intervention; a share link expires and revokes |

Sequencing rationale: phase 1 closes the "anyone can publish as anyone" hole with zero
new services; phase 2 makes it operable at fleet scale; 3 makes identity cryptographic;
4–5 are pure platform UX on top of a then-solid identity layer.
