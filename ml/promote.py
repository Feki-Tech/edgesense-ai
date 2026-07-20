"""Champion/challenger promotion gate for the EdgeSense anomaly model.

Scores a candidate bundle on fresh held-out data (healthy + fault, generated
with a different seed than training) using the same hybrid scorer that runs
in production, then decides whether it is fit to replace the champion:

- absolute floor: false-positive rate <= --max-fp AND detection rate >= --min-tp
- relative bar (only if a champion bundle is given): no worse than the champion
  by more than --tolerance on either metric

Exit code 0 means "promote", 1 means "keep as challenger" — so it composes in
shell:  python ml/promote.py && python register_model.py --promote

Usage:
    python ml/promote.py [--candidate ml/model] [--champion path/to/bundle]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.scoring import score_sample  # noqa: E402
from ml.train import fault_data, normal_data  # noqa: E402

HELDOUT_SEED = 1337  # differs from train.py's 42: the gate never sees training data


def _load_bundle(path: Path) -> dict:
    if path.is_dir():
        path = path / "model.joblib"
    return joblib.load(path)


def _score(bundle: dict, x_normal: np.ndarray, x_fault: np.ndarray) -> tuple[float, float]:
    def rate(x: np.ndarray) -> float:
        return float(np.mean([score_sample(bundle, row)[1] for row in x]))

    return rate(x_normal), rate(x_fault)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=str(Path(__file__).parent / "model"),
                    help="Candidate bundle: ml/model/ dir or a model.joblib path")
    ap.add_argument("--champion", default=None,
                    help="Current champion bundle to beat (optional)")
    ap.add_argument("--max-fp", type=float, default=0.02,
                    help="Max false-positive rate on healthy data")
    ap.add_argument("--min-tp", type=float, default=0.95,
                    help="Min detection rate on fault data")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="Allowed regression vs champion on either metric")
    ap.add_argument("--samples", type=int, default=5_000,
                    help="Held-out samples per class")
    args = ap.parse_args()

    rng = np.random.default_rng(HELDOUT_SEED)
    x_normal = normal_data(args.samples, rng)
    x_fault = fault_data(args.samples, rng)

    candidate = _load_bundle(Path(args.candidate))
    fp, tp = _score(candidate, x_normal, x_fault)
    print(f"candidate: false-positive rate = {fp:.3%}, detection rate = {tp:.3%}")

    if fp > args.max_fp or tp < args.min_tp:
        print(f"GATE FAIL: outside absolute limits (fp <= {args.max_fp:.1%}, "
              f"tp >= {args.min_tp:.1%})")
        return 1

    if args.champion:
        champ_fp, champ_tp = _score(_load_bundle(Path(args.champion)), x_normal, x_fault)
        print(f"champion:  false-positive rate = {champ_fp:.3%}, detection rate = {champ_tp:.3%}")
        if fp > champ_fp + args.tolerance or tp < champ_tp - args.tolerance:
            print(f"GATE FAIL: regresses vs champion beyond tolerance ({args.tolerance:.1%})")
            return 1

    print("GATE PASS: candidate is fit for promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
