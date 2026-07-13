"""Offline evaluation of the anomaly model against simulated fault episodes.

Reuses the simulator's physics (the exact generative process of the live
demo) to create labeled fault episodes and a long healthy run, scores every
reading with the hybrid scorer, and reports detection quality per fault type:

- episode detection rate (was the episode caught at all)
- detection latency (readings / seconds until the first alarm)
- reading-level recall and trigger reasons (model / limit / both)
- false-positive rate on healthy data

Usage:
    python ml/evaluate.py [--model ml/model/model.joblib] [--out docs/EVALUATION.md]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import time
from collections import Counter
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.scoring import score_sample  # noqa: E402
from simulator.simulate import FAULT_TYPES, Machine  # noqa: E402

READING_INTERVAL_S = 0.5  # simulator default publish interval


def _episode_readings(fault: str, ticks: int, seed: int) -> list[dict]:
    """Generate one fault episode; every returned reading is labeled faulty."""
    machine = Machine(machine_id="eval", rng=random.Random(seed))
    readings: list[dict] = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(5):  # warm-up, discarded
            machine.step(anomaly_prob=0.0)
        machine.start_fault(fault, ticks)
        while machine.fault is not None:
            readings.append(machine.step(anomaly_prob=0.0))
    return readings


def evaluate(bundle: dict, *, episodes: int = 25, ticks: int = 30,
             healthy: int = 20_000, seed: int = 7) -> dict:
    features = bundle["features"]

    def flag(reading: dict) -> tuple[bool, str | None]:
        _, is_anomaly, reason = score_sample(bundle, [reading[f] for f in features])
        return is_anomaly, reason

    results: dict = {"faults": {}, "healthy": {}}

    for fi, fault in enumerate(FAULT_TYPES):
        latencies: list[int] = []
        detected = 0
        hits = total = 0
        reasons: Counter = Counter()
        for ep in range(episodes):
            readings = _episode_readings(fault, ticks, seed=seed * 1_000 + fi * 101 + ep)
            first_hit: int | None = None
            for i, r in enumerate(readings):
                is_anomaly, reason = flag(r)
                total += 1
                if is_anomaly:
                    hits += 1
                    reasons[reason] += 1
                    if first_hit is None:
                        first_hit = i
            if first_hit is not None:
                detected += 1
                latencies.append(first_hit)
        latencies.sort()
        results["faults"][fault] = {
            "episodes": episodes,
            "detected": detected,
            "episode_rate": detected / episodes,
            "median_latency_readings": latencies[len(latencies) // 2] if latencies else None,
            "p90_latency_readings": latencies[int(len(latencies) * 0.9)] if latencies else None,
            "reading_recall": hits / total if total else 0.0,
            "reasons": dict(reasons),
        }

    machine = Machine(machine_id="eval-healthy", rng=random.Random(seed))
    fp = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(healthy):
            is_anomaly, _ = flag(machine.step(anomaly_prob=0.0))
            fp += is_anomaly
    results["healthy"] = {"n": healthy, "fp": fp, "fp_rate": fp / healthy}

    return results


def _fmt_latency(readings: int | None) -> str:
    if readings is None:
        return "—"
    n = readings + 1
    return f"{n} reading{'s' if n != 1 else ''} (~{n * READING_INTERVAL_S:.1f}s)"


def to_markdown(results: dict, meta: dict) -> str:
    lines = [
        "# EdgeSense AI — model evaluation",
        "",
        f"- model: `{meta['model']}`",
        f"- {meta['episodes']} episodes per fault type × {meta['ticks']} ticks, "
        f"{results['healthy']['n']:,} healthy readings, seed {meta['seed']}",
        f"- generated: {meta['date']} by `ml/evaluate.py`",
        "",
        "## Fault detection",
        "",
        "| Fault | Episodes detected | Median time-to-detect | p90 time-to-detect "
        "| Reading recall | Trigger reasons |",
        "|---|---|---|---|---|---|",
    ]
    for fault, r in results["faults"].items():
        reasons = ", ".join(f"{k}: {v}" for k, v in
                            sorted(r["reasons"].items(), key=lambda kv: -kv[1])) or "—"
        lines.append(
            f"| {fault} | {r['detected']}/{r['episodes']} ({r['episode_rate']:.0%}) "
            f"| {_fmt_latency(r['median_latency_readings'])} "
            f"| {_fmt_latency(r['p90_latency_readings'])} "
            f"| {r['reading_recall']:.0%} | {reasons} |")

    h = results["healthy"]
    if h["fp"]:
        alarm_every = f"{h['n'] / h['fp'] * READING_INTERVAL_S / 60:.0f} min"
    else:
        alarm_every = "∞"
    lines += [
        "",
        "## Healthy operation",
        "",
        f"- false positives: **{h['fp']} / {h['n']:,} readings "
        f"({h['fp_rate']:.2%})** — at 2 Hz that is one false alarm every "
        f"{alarm_every} per machine",
        "",
        "## Reading the numbers",
        "",
        "- *time-to-detect* counts sensor readings from fault onset to the first alarm"
        f" ({READING_INTERVAL_S}s apart at the simulator's default rate).",
        "- *reading recall* is per-reading; early-episode readings can be subtle, so"
        " recall < 100% while every episode is still caught.",
        "- *trigger reasons*: `model` = IsolationForest, `limit` = z-score guard"
        " (> 6σ on a single feature), `model+limit` = both agreed.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ml/model/model.joblib")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--healthy", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    results = evaluate(bundle, episodes=args.episodes, ticks=args.ticks,
                       healthy=args.healthy, seed=args.seed)
    md = to_markdown(results, {
        "model": args.model, "episodes": args.episodes, "ticks": args.ticks,
        "seed": args.seed, "date": time.strftime("%Y-%m-%d"),
    })
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"(written to {args.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
