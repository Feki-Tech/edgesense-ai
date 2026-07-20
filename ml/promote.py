"""Champion/challenger promotion gate (MLOps phase 1).

Trains a *challenger* bundle, replays the offline evaluation harness
(``ml/evaluate.py`` — the exact simulator physics of the live demo) on both
the challenger and the current *champion* (``ml/model/model.joblib``, if one
exists), and only promotes the challenger when it

1. clears the absolute quality bar — every fault episode detected, ~1-reading
   median time-to-detect, healthy false-positive rate within budget,
2. passes the ONNX parity check (onnxruntime reproduces the numpy scorer), and
3. is no worse than the champion — episodes detected, per-fault median
   time-to-detect and healthy FP rate, within small tolerances.

Promotion atomically replaces the champion's ``model.joblib`` +
``model.manifest.json`` + ``MODEL_CARD.md`` (via ``ml/manifest.save_bundle``);
refusal prints a diff table and leaves the champion untouched. The candidate
bundle, manifest, model card and the gate report are always written to
``--out-dir`` so CI can archive them either way.

Exit codes: 0 promoted · 1 refused · 2 error.

    python ml/promote.py                     # sklearn challenger vs champion
    python ml/promote.py --backend torch     # PyTorch challenger (local only)
    make promote / make promote-torch
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.evaluate import evaluate  # noqa: E402
from ml.manifest import manifest_path, render_model_card, save_bundle  # noqa: E402
from ml.train import _BACKENDS, MODEL_PATH, fault_data, normal_data, train_autoencoder  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).parent / "model" / "candidate"


@dataclass(frozen=True)
class QualityBar:
    """Absolute gate every promoted model must clear (see docs/EVALUATION.md)."""
    episode_rate: float = 1.0        # every episode of every fault detected
    max_median_latency: int = 1      # 0-based first-hit index; 0 == "1 reading"
    max_fp_rate: float = 0.006       # 0.5% budget + calibration slack
    onnx_rel_mae: float = 1e-3
    onnx_agreement: float = 0.99


# challenger-vs-champion tolerances (the challenger must be *no worse than*)
FP_TOLERANCE = 0.001        # healthy FP rate may exceed the champion's by 0.1pp
LATENCY_TOLERANCE = 1       # readings of extra median time-to-detect per fault


def summarize(results: dict) -> dict:
    """Flatten an evaluate() result into the numbers the gate compares."""
    faults = results["faults"]
    return {
        "episodes_total": sum(r["episodes"] for r in faults.values()),
        "episodes_detected": sum(r["detected"] for r in faults.values()),
        "fp_rate": results["healthy"]["fp_rate"],
        "per_fault": {
            fault: {
                "detected": r["detected"],
                "episodes": r["episodes"],
                "episode_rate": r["episode_rate"],
                "median_latency": r["median_latency_readings"],
                "reading_recall": r["reading_recall"],
            } for fault, r in faults.items()
        },
    }


def check_bar(summary: dict, bar: QualityBar) -> list[str]:
    """Absolute quality-bar failures for one evaluated bundle ([] = pass)."""
    failures = []
    for fault, r in summary["per_fault"].items():
        if r["episode_rate"] < bar.episode_rate:
            failures.append(f"{fault}: {r['detected']}/{r['episodes']} episodes "
                            f"detected (need {bar.episode_rate:.0%})")
        med = r["median_latency"]
        if med is None or med > bar.max_median_latency:
            shown = "—" if med is None else med + 1
            failures.append(f"{fault}: median time-to-detect {shown} readings "
                            f"(bar: ≤ {bar.max_median_latency + 1})")
    if summary["fp_rate"] > bar.max_fp_rate:
        failures.append(f"healthy FP rate {summary['fp_rate']:.3%} "
                        f"(bar: ≤ {bar.max_fp_rate:.3%})")
    return failures


def compare_to_champion(challenger: dict, champion: dict) -> list[str]:
    """Regressions of the challenger vs the champion summary ([] = pass)."""
    failures = []
    if challenger["episodes_detected"] < champion["episodes_detected"]:
        failures.append(
            f"episodes detected {challenger['episodes_detected']} < "
            f"champion's {champion['episodes_detected']}")
    if challenger["fp_rate"] > champion["fp_rate"] + FP_TOLERANCE:
        failures.append(
            f"healthy FP rate {challenger['fp_rate']:.3%} > champion's "
            f"{champion['fp_rate']:.3%} + {FP_TOLERANCE:.1%} tolerance")
    for fault, ch in challenger["per_fault"].items():
        cp = champion["per_fault"].get(fault)
        if cp is None or cp["median_latency"] is None:
            continue
        if ch["median_latency"] is None or \
                ch["median_latency"] > cp["median_latency"] + LATENCY_TOLERANCE:
            shown = "—" if ch["median_latency"] is None else ch["median_latency"] + 1
            failures.append(
                f"{fault}: median time-to-detect {shown} readings > champion's "
                f"{cp['median_latency'] + 1} + {LATENCY_TOLERANCE} tolerance")
    return failures


def onnx_parity(bundle: dict, *, seed: int = 7, n: int = 500) -> tuple[float, float]:
    """(relative score MAE, label agreement) of onnxruntime vs the numpy scorer."""
    import onnxruntime as ort

    from ml.export_onnx import build_onnx
    from ml.scoring import reconstruction_errors

    sess = ort.InferenceSession(build_onnx(bundle).SerializeToString(),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    x = np.vstack([normal_data(n, rng), fault_data(n, rng)]).astype(np.float32)
    outputs = sess.run(None, {sess.get_inputs()[0].name: x})
    named = {o.name: np.ravel(v) for o, v in zip(sess.get_outputs(), outputs)}

    ref_scores = reconstruction_errors(bundle, x)
    ref_labels = (ref_scores > bundle["threshold"]).astype(np.int64)
    rel_mae = float(np.mean(np.abs(named["scores"] - ref_scores)
                            / np.maximum(ref_scores, 1e-6)))
    agreement = float(np.mean(named["label"] == ref_labels))
    return rel_mae, agreement


def _fmt_med(median_latency: int | None) -> str:
    return "—" if median_latency is None else f"{median_latency + 1}"


def render_report(*, challenger_version: str, champion_version: str | None,
                  challenger: dict, champion: dict | None,
                  parity: tuple[float, float], failures: list[str],
                  promoted: bool, bar: QualityBar) -> str:
    """Markdown gate report: side-by-side table + verdict."""
    lines = [
        "# EdgeSense AI — promotion gate report",
        "",
        f"- challenger: `{challenger_version}`",
        f"- champion:  `{champion_version}`" if champion_version
        else "- champion:  *(none — absolute bar only)*",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')} by `ml/promote.py`",
        "",
        "| Metric | Champion | Challenger | Bar |",
        "|---|---|---|---|",
    ]

    def cell(summary: dict | None, fn) -> str:
        return fn(summary) if summary is not None else "—"

    def episodes_cell(s: dict) -> str:
        return f"{s['episodes_detected']}/{s['episodes_total']}"

    def fp_cell(s: dict) -> str:
        return f"{s['fp_rate']:.3%}"

    lines.append(
        "| episodes detected "
        f"| {cell(champion, episodes_cell)} "
        f"| {episodes_cell(challenger)} "
        f"| {bar.episode_rate:.0%} per fault |")
    for fault, ch in challenger["per_fault"].items():
        cp = champion["per_fault"].get(fault) if champion else None
        lines.append(
            f"| {fault} median time-to-detect (readings) "
            f"| {_fmt_med(cp['median_latency']) if cp else '—'} "
            f"| {_fmt_med(ch['median_latency'])} "
            f"| ≤ {bar.max_median_latency + 1} |")
    lines.append(
        "| healthy FP rate "
        f"| {cell(champion, fp_cell)} "
        f"| {fp_cell(challenger)} "
        f"| ≤ {bar.max_fp_rate:.3%} |")
    rel_mae, agreement = parity
    lines.append(f"| ONNX parity (rel MAE / label agreement) | — "
                 f"| {rel_mae:.2e} / {agreement:.2%} "
                 f"| < {bar.onnx_rel_mae:.0e} / > {bar.onnx_agreement:.0%} |")

    lines += ["", "## Verdict", ""]
    if promoted:
        lines.append("**PROMOTED** — the challenger cleared the quality bar"
                     + (" and did not regress vs the champion."
                        if champion_version else " (no champion to beat)."))
    else:
        lines.append("**REFUSED** — the champion keeps serving:")
        lines += [""] + [f"- {f}" for f in failures]
    return "\n".join(lines) + "\n"


def run_gate(*, backend: str = "sklearn", seed: int = 42, epochs: int | None = None,
             champion_path: "Path | str" = MODEL_PATH,
             out_dir: "Path | str" = DEFAULT_OUT_DIR,
             bar: QualityBar = QualityBar(),
             episodes: int = 25, ticks: int = 30, healthy: int = 20_000,
             eval_seed: int = 7, n_train: int = 20_000, n_cal: int = 20_000,
             ) -> tuple[int, str]:
    """Train, evaluate, compare and (maybe) promote. Returns (exit_code, report)."""
    champion_path = Path(champion_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_kwargs = dict(episodes=episodes, ticks=ticks, healthy=healthy, seed=eval_seed)

    fit_epochs = epochs or _BACKENDS[backend][1]
    print(f"training challenger ({backend} backend, seed {seed}, "
          f"epochs {fit_epochs}) ...")
    challenger_bundle = train_autoencoder(backend, seed=seed, epochs=fit_epochs,
                                          n_train=n_train, n_cal=n_cal)
    challenger_version = challenger_bundle["manifest"]["model_version"]

    print(f"evaluating challenger {challenger_version} "
          f"({episodes} episodes/fault, {healthy:,} healthy) ...")
    challenger = summarize(evaluate(challenger_bundle, **eval_kwargs))

    champion_bundle = champion_version = champion = None
    if champion_path.exists():
        champion_bundle = joblib.load(champion_path)
        champion_version = (champion_bundle.get("manifest") or {}).get(
            "model_version", "unknown")
        print(f"evaluating champion {champion_version} ({champion_path}) ...")
        champion = summarize(evaluate(champion_bundle, **eval_kwargs))
    else:
        print(f"no champion at {champion_path} — absolute bar only")

    print("checking ONNX parity ...")
    parity = onnx_parity(challenger_bundle, seed=eval_seed)

    failures = check_bar(challenger, bar)
    rel_mae, agreement = parity
    if rel_mae >= bar.onnx_rel_mae or agreement <= bar.onnx_agreement:
        failures.append(f"ONNX parity: rel MAE {rel_mae:.2e} / agreement "
                        f"{agreement:.2%} (bar: < {bar.onnx_rel_mae:.0e} / "
                        f"> {bar.onnx_agreement:.0%})")
    if champion is not None:
        failures += compare_to_champion(challenger, champion)
    promoted = not failures

    # snapshot the eval metrics into the challenger manifest before writing it
    challenger_bundle["manifest"]["metrics"].update({
        "eval_episodes_detected":
            f"{challenger['episodes_detected']}/{challenger['episodes_total']}",
        "eval_fp_rate": round(challenger["fp_rate"], 5),
        "eval_median_latency_readings": {
            fault: (None if r["median_latency"] is None else r["median_latency"] + 1)
            for fault, r in challenger["per_fault"].items()},
        "onnx_rel_mae": rel_mae,
        "onnx_label_agreement": agreement,
    })

    report = render_report(challenger_version=challenger_version,
                           champion_version=champion_version,
                           challenger=challenger, champion=champion,
                           parity=parity, failures=failures, promoted=promoted,
                           bar=bar)

    # candidate artifacts are always archived, promoted or not
    save_bundle(challenger_bundle, out_dir / "model.joblib")
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    if promoted:
        save_bundle(challenger_bundle, champion_path)  # atomic os.replace inside
        print(f"\npromoted {challenger_version} -> {champion_path} "
              "(+ manifest + model card)")
    print(f"\n{report}")
    print(f"candidate artifacts -> {out_dir}")
    return (0 if promoted else 1), report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=tuple(_BACKENDS), default="sklearn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (default: 500 sklearn / 300 torch)")
    ap.add_argument("--model", default=str(MODEL_PATH),
                    help="champion bundle path (also the promotion target)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="candidate artifact directory (bundle, manifest, "
                         "model card, report)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--healthy", type=int, default=20_000)
    ap.add_argument("--eval-seed", type=int, default=7)
    args = ap.parse_args()

    # never let console encoding (e.g. Windows cp1252) fail a completed gate
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        code, _ = run_gate(backend=args.backend, seed=args.seed, epochs=args.epochs,
                           champion_path=args.model, out_dir=args.out_dir,
                           episodes=args.episodes, ticks=args.ticks,
                           healthy=args.healthy, eval_seed=args.eval_seed)
        return code
    except Exception as exc:  # noqa: BLE001 - CI needs a distinct error code
        print(f"promotion gate error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
