# EdgeSense AI — rotating-machine benchmark (real feature contract)

Same architecture and calibration as the shipped model (`ml/train.py`),
trained on the healthy readings of the [KAIST rotating-machine dataset](https://data.mendeley.com/datasets/ztmf3m7h5x) (CC BY 4.0) reduced to EdgeSense's own reading contract — `{vibration, temperature, current}` at 2 Hz, from 25.6 kHz raw.

Unlike [BENCHMARK.md](BENCHMARK.md) (AI4I 2020, five unrelated tabular
features) this exercises the three features the sidecar actually serves.

- backend: `sklearn` · 648 healthy training readings, 216 calibration · threshold 0.00977 (99.5% healthy quantile)
- generated: 2026-08-12 by `ml/benchmark_rotating.py` (seed 7)

## Fault detection

| Condition | Fault | Readings | ROC-AUC | Recall @ 0.5% FP | Hybrid (+6σ) |
|---|---|---|---|---|---|
| BPFI_03 | bearing inner race fault | 360 | 1.000 | 100% | 100% |
| BPFI_10 | bearing inner race fault | 360 | 1.000 | 100% | 100% |
| BPFI_30 | bearing inner race fault | 360 | 1.000 | 100% | 100% |
| BPFO_03 | bearing outer race fault | 360 | 1.000 | 100% | 100% |
| BPFO_10 | bearing outer race fault | 360 | 1.000 | 100% | 100% |
| BPFO_30 | bearing outer race fault | 360 | 1.000 | 100% | 100% |
| Misalign_01 | shaft misalignment | 720 | 1.000 | 100% | 100% |
| Misalign_03 | shaft misalignment | 720 | 1.000 | 100% | 100% |
| Misalign_05 | shaft misalignment | 720 | 1.000 | 100% | 100% |
| Unbalance_0583mg | rotor unbalance | 480 | 0.917 | 50% | 50% |
| Unbalance_1169mg | rotor unbalance | 480 | 0.803 | 50% | 50% |
| Unbalance_1751mg | rotor unbalance | 480 | 0.892 | 50% | 50% |
| Unbalance_2239mg | rotor unbalance | 480 | 0.938 | 50% | 50% |
| Unbalance_3318mg | rotor unbalance | 480 | 0.976 | 51% | 51% |

- overall ROC-AUC of the reconstruction error (healthy test vs all faults): **0.966**
- false positives on 216 held-out healthy readings: model **0.00%**, hybrid **0.00%**
- Healthy readings from all three load levels (0/2/4 Nm) are pooled: load is an operating condition, not a fault, so the baseline has to cover all of them.
- Windows are non-overlapping, so no millisecond appears in both the training and the test split.
- Recall is per *reading* at a strict 0.5%-FP operating point. EdgeSense's streaming setting is more forgiving: an episode counts as caught if any reading in it trips (see [EVALUATION.md](EVALUATION.md)).
- Severity is encoded in the condition name (`BPFI_03` = 0.3 mm inner race, `_30` = 3.0 mm; `Unbalance_0583mg` … `_3318mg`), so the table doubles as a sensitivity curve — detection should rise with severity.
