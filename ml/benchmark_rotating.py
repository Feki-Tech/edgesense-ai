"""Benchmark the EdgeSense autoencoder on real vibration/temperature/current data.

Companion to `ml/benchmark_public.py`. That one trains on AI4I 2020, which is
real but exposes five unrelated tabular features (air/process temperature,
rotational speed, torque, tool wear) — it validates the *architecture*, not the
contract EdgeSense actually serves. This one trains the same architecture on
the KAIST rotating-machine dataset reduced to the real reading contract,
`{vibration, temperature, current}` at ~2 Hz, so the numbers speak to the
shipped feature set.

    make benchmark-rotating
    python ml/benchmark_rotating.py --data ml/data/rotating [--backend torch]

The dataset is ~4.3 GB and is NOT downloaded automatically — see
docs/BENCHMARK.md for the one-time fetch. Everything downstream of
`ml/rotating.py` is the production code path: same `build_bundle`, same
threshold calibration, same hybrid scoring.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.rotating import (READING_RATE_HZ, SAMPLE_RATE_HZ, DatasetError,  # noqa: E402
                         FEATURES, load_dataset, parse_stem)
from ml.scoring import reconstruction_errors  # noqa: E402
from ml.train import build_bundle  # noqa: E402

DATA_ROOT = Path(__file__).parent / "data" / "rotating"

DOWNLOAD_HINT = """\
Dataset not found at {root}.

It is ~4.3 GB and CC BY 4.0, so it is fetched once by hand rather than by this
script. From https://data.mendeley.com/datasets/ztmf3m7h5x download:

    vibration.zip      (2.7 GB)
    current,temp.zip   (1.5 GB)     # acoustic.zip is not needed

and unpack both into {root}/ so it contains:

    {root}/vibration/0Nm_Normal.mat ...
    {root}/current,temp/0Nm_Normal.tdms ...

See docs/BENCHMARK.md."""


def run_benchmark(healthy: np.ndarray, faults: "dict[str, np.ndarray]",
                  backend: str = "sklearn", *, seed: int = 7,
                  epochs: int | None = None) -> dict:
    """Train on healthy readings (60/20/20 split), score every fault condition."""
    from sklearn.metrics import roc_auc_score

    idx = np.random.default_rng(seed).permutation(len(healthy))
    n_train, n_cal = int(len(healthy) * 0.6), int(len(healthy) * 0.2)
    x_train = healthy[idx[:n_train]]
    x_cal = healthy[idx[n_train:n_train + n_cal]]
    x_test = healthy[idx[n_train + n_cal:]]

    bundle = build_bundle(x_train, x_cal, FEATURES, backend, seed=seed, epochs=epochs)
    thr, guard = bundle["threshold"], bundle["z_guard"]

    def score(rows: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """(model verdicts, hybrid verdicts) — mirrors ml/scoring.py, vectorised."""
        errs = reconstruction_errors(bundle, rows)
        z = np.abs((rows - bundle["scaler_mean"]) / bundle["scaler_scale"])
        return errs > thr, (errs > thr) | (z.max(axis=1) > guard)

    model_fp, hybrid_fp = (float(np.mean(v)) for v in score(x_test))
    errs_healthy = reconstruction_errors(bundle, x_test)

    results: dict = {
        "healthy": {"n": len(x_test), "model_fp": model_fp, "hybrid_fp": hybrid_fp},
        "conditions": {},
        "bundle": {"backend": backend, "threshold": float(thr),
                   "n_train": len(x_train), "n_cal": len(x_cal)},
    }

    all_errs = []
    for condition, rows in faults.items():
        model_hit, hybrid_hit = score(rows)
        errs = reconstruction_errors(bundle, rows)
        results["conditions"][condition] = {
            # every condition appears at each load, so any stem parses the family
            "family": parse_stem(f"0Nm_{condition}").family,
            "n": len(rows),
            "auc": float(roc_auc_score(
                np.r_[np.zeros(len(errs_healthy)), np.ones(len(errs))],
                np.r_[errs_healthy, errs])),
            "model_recall": float(np.mean(model_hit)),
            "hybrid_recall": float(np.mean(hybrid_hit)),
        }
        all_errs.append(errs)

    errs_all = np.concatenate([errs_healthy, *all_errs])
    labels_all = np.concatenate([np.zeros(len(x_test)),
                                 np.ones(sum(len(e) for e in all_errs))])
    results["auc"] = float(roc_auc_score(labels_all, errs_all))
    return results


def to_markdown(results: dict, meta: dict) -> str:
    b, h = results["bundle"], results["healthy"]
    lines = [
        "# EdgeSense AI — rotating-machine benchmark (real feature contract)",
        "",
        "Same architecture and calibration as the shipped model (`ml/train.py`),",
        "trained on the healthy readings of the [KAIST rotating-machine"
        " dataset](https://data.mendeley.com/datasets/ztmf3m7h5x) (CC BY 4.0)"
        " reduced to EdgeSense's own reading contract —"
        " `{vibration, temperature, current}` at"
        f" {meta['reading_rate']:g} Hz, from {meta['sample_rate'] / 1000:g} kHz raw.",
        "",
        "Unlike [BENCHMARK.md](BENCHMARK.md) (AI4I 2020, five unrelated tabular",
        "features) this exercises the three features the sidecar actually serves.",
        "",
        f"- backend: `{b['backend']}` · {b['n_train']:,} healthy training readings, "
        f"{b['n_cal']:,} calibration · threshold {b['threshold']:.5f} "
        f"(99.5% healthy quantile)",
        f"- generated: {meta['date']} by `ml/benchmark_rotating.py` (seed {meta['seed']})",
        "",
        "## Fault detection",
        "",
        "| Condition | Fault | Readings | ROC-AUC | Recall @ 0.5% FP | Hybrid (+6σ) |",
        "|---|---|---|---|---|---|",
    ]
    for cond, r in results["conditions"].items():
        lines.append(f"| {cond} | {r['family']} | {r['n']} | {r['auc']:.3f} "
                     f"| {r['model_recall']:.0%} | {r['hybrid_recall']:.0%} |")
    lines += [
        "",
        f"- overall ROC-AUC of the reconstruction error (healthy test vs all"
        f" faults): **{results['auc']:.3f}**",
        f"- false positives on {h['n']:,} held-out healthy readings: "
        f"model **{h['model_fp']:.2%}**, hybrid **{h['hybrid_fp']:.2%}**",
        "- Healthy readings from all three load levels (0/2/4 Nm) are pooled:"
        " load is an operating condition, not a fault, so the baseline has to"
        " cover all of them.",
        "- Windows are non-overlapping, so no millisecond appears in both the"
        " training and the test split.",
        "- Recall is per *reading* at a strict 0.5%-FP operating point."
        " EdgeSense's streaming setting is more forgiving: an episode counts as"
        " caught if any reading in it trips (see [EVALUATION.md](EVALUATION.md)).",
        "- Severity is encoded in the condition name (`BPFI_03` = 0.3 mm inner"
        " race, `_30` = 3.0 mm; `Unbalance_0583mg` … `_3318mg`), so the table"
        " doubles as a sensitivity curve — detection should rise with severity.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA_ROOT), help="dataset root")
    ap.add_argument("--backend", choices=("sklearn", "torch"), default="sklearn")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-rate", type=float, default=SAMPLE_RATE_HZ)
    ap.add_argument("--reading-rate", type=float, default=READING_RATE_HZ,
                    help="readings per second after reduction (the ~2 Hz contract)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    # The report contains σ (the z-guard); a Windows console defaults to cp1252
    # and print() would raise UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = Path(args.data)
    if not root.exists():
        print(DOWNLOAD_HINT.format(root=root), file=sys.stderr)
        return 2

    try:
        healthy, faults = load_dataset(root, sample_rate=args.sample_rate,
                                       reading_rate=args.reading_rate, progress=True)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{len(healthy):,} healthy readings, "
          f"{sum(len(v) for v in faults.values()):,} fault readings "
          f"across {len(faults)} conditions", file=sys.stderr)

    results = run_benchmark(healthy, faults, args.backend, seed=args.seed,
                            epochs=args.epochs)
    md = to_markdown(results, {"seed": args.seed, "date": time.strftime("%Y-%m-%d"),
                               "sample_rate": args.sample_rate,
                               "reading_rate": args.reading_rate})
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"(written to {args.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
