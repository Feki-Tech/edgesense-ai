# Public datasets

EdgeSense trains on synthetic data by default (`simulator/simulate.py` →
`ml/train.py`). These are the real, openly licensed datasets used to check that
the model is not just learning its own simulator.

Both benchmark reports are **generated** — `docs/BENCHMARK.md` and
`docs/BENCHMARK-ROTATING.md` are overwritten by `make benchmark` /
`make benchmark-rotating`. This file is the stable one; put acquisition notes
here, not in the reports.

| Dataset | Features | Matches the reading contract? | Size | License |
|---|---|---|---|---|
| [AI4I 2020](https://archive.ics.uci.edu/dataset/601) | air/process temperature, rotational speed, torque, tool wear | ✗ five unrelated tabular features | 0.5 MB | CC BY 4.0 |
| [KAIST rotating machine](https://data.mendeley.com/datasets/ztmf3m7h5x) | vibration, temperature, motor current (+ acoustic) | ✓ all three, same machine, same time | 4.3 GB | CC BY 4.0 |

## Why two

AI4I is small, downloads automatically, and is a fine architecture check — but
none of its columns are vibration or current, so it never exercised the
contract the sidecar actually serves (`{machine_id, ts, vibration, temperature,
current}`). The KAIST set is the only openly licensed one carrying all three
signals simultaneously, which is what makes the second benchmark worth its
download size.

> Considered and rejected: the [Paderborn KAt bearing
> dataset](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter) has
> synchronised current + vibration and *real* accelerated-life damage rather
> than seeded faults — better evidence in principle — but it is **CC BY-NC
> 4.0**, noncommercial only. KAIST is CC BY 4.0 and imposes no such limit.

## AI4I 2020 — automatic

```bash
make benchmark          # downloads ~0.5 MB into ml/data/ (gitignored) on first run
```

## KAIST rotating machine — one-time manual fetch

4.3 GB is too much to pull silently from a `make` target, so it is deliberately
not automatic. From <https://data.mendeley.com/datasets/ztmf3m7h5x> download:

- `vibration.zip` (2.7 GB)
- `current,temp.zip` (1.5 GB)

`acoustic.zip` is **not** needed — EdgeSense has no microphone in its contract.

Unpack both into `ml/data/rotating/` so it looks like:

```
ml/data/rotating/
├── vibration/            0Nm_Normal.mat, 0Nm_BPFI_03.mat, … (45 files)
└── current,temp/         0Nm_Normal.tdms, 0Nm_BPFI_03.tdms, … (45 files)
```

A flat layout (both archives unpacked into one directory) also works. Then:

```bash
uv sync --extra benchmark      # npTDMS + h5py + pandas
make benchmark-rotating        # -> docs/BENCHMARK-ROTATING.md
```

### What the reduction does

Raw signals are sampled at 25.6 kHz; EdgeSense reports at ~2 Hz. `ml/rotating.py`
collapses each 0.5 s window into one reading — exactly the sensor-adapter
behaviour `docs/HARDWARE.md` §4 specifies for real hardware, where 2 Hz is the
*reporting* rate and vibration is sampled far faster and reduced to an RMS
window:

| Contract feature | Reduction | Raw channels |
|---|---|---|
| `vibration` | RMS over the window, pooled across axes | `x/y_direction_housing_A`, `x/y_direction_housing_B` (g) |
| `temperature` | window mean (slow-varying) | `Temperature_housing_A`, `Temperature_housing_B` (°C) |
| `current` | per-phase RMS, then averaged | `U-phase`, `V-phase`, `W-phase` (A) |

Windows are **non-overlapping**: overlapping them would inflate the sample count
while leaking the same milliseconds into both the training and test splits.

Healthy readings from all three load levels (0/2/4 Nm) are pooled — load is an
operating condition, not a fault, so the baseline has to span all of them.

### Units differ from the simulator, on purpose

The simulator emits vibration as mm/s RMS *velocity*; this dataset measures
acceleration in g. No conversion is attempted. The benchmark trains its own
bundle on the dataset's own units (exactly as `benchmark_public.py` does with
AI4I), so the question it answers is "does this architecture separate healthy
from faulty on real three-signal data", not "does the shipped model score this
rig". Reusing the shipped bundle would require an integration step and a
calibration that nothing here justifies.

### Conditions

15 machine states per load level, 3 load levels:

- `Normal` — 120 s per load
- `BPFI_{03,10,30}` — bearing inner race, 0.3 / 1.0 / 3.0 mm
- `BPFO_{03,10,30}` — bearing outer race, same severities
- `Misalign_{01,03,05}` — shaft misalignment, 0.1 / 0.3 / 0.5 mm
- `Unbalance_{0583,1169,1751,2239,3318}mg` — rotor unbalance
- every fault: 60 s per load

Severity is encoded in the name, so the report doubles as a sensitivity curve.

## Citation

> W. Jung, S.-H. Yun, J. Bae, D. Lim, S.-H. Kim, Y.-H. Park, "Vibration,
> acoustic, temperature, and motor current dataset of rotating machine under
> varying operating conditions for fault diagnosis", *Data in Brief*, 2023.
> Mendeley Data, doi:10.17632/ztmf3m7h5x.1 (CC BY 4.0)

> S. Matzka, "AI4I 2020 Predictive Maintenance Dataset", UCI Machine Learning
> Repository, 2020. <https://archive.ics.uci.edu/dataset/601> (CC BY 4.0)
