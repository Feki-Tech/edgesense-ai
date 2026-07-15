# EdgeSense AI — model evaluation

- model: `ml/model/model.joblib`
- 25 episodes per fault type × 30 ticks, 20,000 healthy readings, seed 7
- generated: 2026-07-15 by `ml/evaluate.py`

## Fault detection

| Fault | Episodes detected | Median time-to-detect | p90 time-to-detect | Reading recall | Trigger reasons |
|---|---|---|---|---|---|
| bearing_fault | 25/25 (100%) | 1 reading (~0.5s) | 1 reading (~0.5s) | 100% | model+limit: 743, model: 6 |
| overheat | 25/25 (100%) | 1 reading (~0.5s) | 1 reading (~0.5s) | 100% | model+limit: 750 |
| overload | 25/25 (100%) | 1 reading (~0.5s) | 1 reading (~0.5s) | 100% | model+limit: 750 |

## Healthy operation

- false positives: **87 / 20,000 readings (0.43%)** — at 2 Hz that is one false alarm every 2 min per machine

## Reading the numbers

- *time-to-detect* counts sensor readings from fault onset to the first alarm (0.5s apart at the simulator's default rate).
- *reading recall* is per-reading; early-episode readings can be subtle, so recall < 100% while every episode is still caught.
- *trigger reasons*: `model` = autoencoder (reconstruction error above its calibrated threshold), `limit` = z-score guard (> 6σ on a single feature), `model+limit` = both agreed.
