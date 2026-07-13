# EdgeSense AI — model evaluation

- model: `ml/model/model.joblib`
- 25 episodes per fault type × 30 ticks, 20,000 healthy readings, seed 7
- generated: 2026-07-13 by `ml/evaluate.py`

## Fault detection

| Fault | Episodes detected | Median time-to-detect | p90 time-to-detect | Reading recall | Trigger reasons |
|---|---|---|---|---|---|
| bearing_fault | 25/25 (100%) | 1 reading (~0.5s) | 1 reading (~0.5s) | 100% | model+limit: 665, limit: 78, model: 5 |
| overheat | 25/25 (100%) | 1 reading (~0.5s) | 1 reading (~0.5s) | 100% | limit: 597, model+limit: 153 |
| overload | 25/25 (100%) | 1 reading (~0.5s) | 1 reading (~0.5s) | 100% | model+limit: 584, limit: 166 |

## Healthy operation

- false positives: **96 / 20,000 readings (0.48%)** — at 2 Hz that is one false alarm every 2 min per machine

## Reading the numbers

- *time-to-detect* counts sensor readings from fault onset to the first alarm (0.5s apart at the simulator's default rate).
- *reading recall* is per-reading; early-episode readings can be subtle, so recall < 100% while every episode is still caught.
- *trigger reasons*: `model` = IsolationForest, `limit` = z-score guard (> 6σ on a single feature), `model+limit` = both agreed.
