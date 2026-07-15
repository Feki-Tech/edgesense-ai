# EdgeSense AI — public-dataset benchmark

Same architecture and calibration as the shipped model (`ml/train.py`),
trained on the healthy rows of the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601) (UCI, CC BY 4.0): 10,000 real-world-modelled milling readings, 5 sensor features, labeled failure modes.

- backend: `sklearn` · 5,796 healthy training rows, 1,932 calibration rows · threshold 1.335 (99.5% healthy quantile)
- generated: 2026-07-15 by `ml/benchmark_public.py` (seed 7)

## Failure detection

| Failure mode | Rows | ROC-AUC | Recall @ 0.5% FP | Hybrid recall (+6σ guard) |
|---|---|---|---|---|
| HDF — heat dissipation failure | 115 | 0.780 | 3% | 3% |
| PWF — power failure | 95 | 0.970 | 37% | 43% |
| OSF — overstrain failure | 98 | 0.920 | 7% | 7% |
| TWF — tool wear failure | 46 | 0.740 | 0% | 0% |

- overall ROC-AUC of the reconstruction error (healthy test vs all failures): **0.864**
- false positives on 1,933 held-out healthy rows: model **0.52%**, hybrid **0.52%**
- Recall here is per *snapshot* at a strict 0.5%-FP operating point — conservative compared to EdgeSense's streaming setting, where an episode is caught if any reading in it trips (see [EVALUATION.md](EVALUATION.md)). AUC shows the threshold-free separability of each failure mode.
- AI4I failure modes are joint-distribution violations (e.g. HDF = small air/process temperature gap *and* low rotational speed, each individually normal) — exactly the regime where the autoencoder adds value over per-feature limits: on its own the 6σ guard only catches the most extreme power-failure spikes (26% of PWF) and misses HDF, OSF and TWF entirely.
- RNF (random failures) is excluded: it has no feature signal by construction and is undetectable from sensor data.
- TWF depends on tool-wear values that healthy rows also reach, so point-wise recall is inherently limited on this mode.
