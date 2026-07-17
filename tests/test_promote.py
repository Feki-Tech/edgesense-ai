"""Champion/challenger promotion gate: decision logic + end-to-end runs."""

from __future__ import annotations

import json

import joblib
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from ml.promote import (QualityBar, check_bar, compare_to_champion,  # noqa: E402
                        render_report, run_gate)

FAULTS = ("bearing_fault", "overheat", "overload")


def _summary(detected=(25, 25, 25), episodes=25, fp=0.004, medians=(0, 0, 0)) -> dict:
    return {
        "episodes_total": episodes * len(FAULTS),
        "episodes_detected": sum(detected),
        "fp_rate": fp,
        "per_fault": {
            fault: {
                "detected": d,
                "episodes": episodes,
                "episode_rate": d / episodes,
                "median_latency": m,
                "reading_recall": 0.9,
            } for fault, d, m in zip(FAULTS, detected, medians)
        },
    }


def test_bar_passes_a_good_model() -> None:
    assert check_bar(_summary(), QualityBar()) == []


def test_bar_catches_missed_episodes() -> None:
    failures = check_bar(_summary(detected=(25, 24, 25)), QualityBar())
    assert len(failures) == 1 and "overheat" in failures[0]


def test_bar_catches_slow_detection_and_none_median() -> None:
    failures = check_bar(_summary(medians=(0, 5, None)), QualityBar())
    assert len(failures) == 2
    assert any("overheat" in f for f in failures)
    assert any("overload" in f for f in failures)


def test_bar_catches_high_fp_rate() -> None:
    failures = check_bar(_summary(fp=0.02), QualityBar())
    assert len(failures) == 1 and "FP rate" in failures[0]


def test_champion_comparison_accepts_equal_or_better() -> None:
    assert compare_to_champion(_summary(), _summary()) == []
    assert compare_to_champion(_summary(fp=0.002), _summary(fp=0.004)) == []


def test_champion_comparison_catches_regressions() -> None:
    champion = _summary()
    fewer = compare_to_champion(_summary(detected=(25, 23, 25)), champion)
    assert any("episodes detected" in f for f in fewer)

    worse_fp = compare_to_champion(_summary(fp=0.02), champion)
    assert any("FP rate" in f for f in worse_fp)

    slower = compare_to_champion(_summary(medians=(0, 4, 0)), champion)
    assert any("overheat" in f for f in slower)


def test_report_renders_verdicts() -> None:
    promoted = render_report(
        challenger_version="20990101.000000+abcdef0", champion_version=None,
        challenger=_summary(), champion=None, parity=(1e-6, 1.0),
        failures=[], promoted=True, bar=QualityBar())
    assert "**PROMOTED**" in promoted
    assert "| Metric | Champion | Challenger | Bar |" in promoted

    refused = render_report(
        challenger_version="v2", champion_version="v1",
        challenger=_summary(fp=0.02), champion=_summary(),
        parity=(1e-6, 1.0), failures=["healthy FP rate too high"],
        promoted=False, bar=QualityBar())
    assert "**REFUSED**" in refused
    assert "- healthy FP rate too high" in refused


# --- end-to-end (tiny knobs so the gate runs in seconds) --------------------

_TINY = dict(seed=0, epochs=150, n_train=4_000, n_cal=2_000,
             episodes=3, ticks=20, healthy=1_500, eval_seed=1)
_TINY_BAR = QualityBar(max_median_latency=3, max_fp_rate=0.03)


@pytest.fixture(scope="module")
def promoted_gate(tmp_path_factory):
    champion = tmp_path_factory.mktemp("champ") / "model.joblib"
    out_dir = tmp_path_factory.mktemp("candidate")
    code, report = run_gate(champion_path=champion, out_dir=out_dir,
                            bar=_TINY_BAR, **_TINY)
    return code, report, champion, out_dir


def test_gate_promotes_without_champion(promoted_gate) -> None:
    code, report, champion, out_dir = promoted_gate
    assert code == 0
    assert "**PROMOTED**" in report

    # champion artifacts written atomically next to the bundle
    assert champion.exists()
    manifest = json.loads((champion.parent / "model.manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["model_version"] == \
        joblib.load(champion)["manifest"]["model_version"]
    assert "eval_fp_rate" in manifest["metrics"]
    assert (champion.parent / "MODEL_CARD.md").exists()

    # candidate artifacts always archived for CI
    for name in ("model.joblib", "model.manifest.json", "MODEL_CARD.md", "report.md"):
        assert (out_dir / name).exists(), name


def test_gate_refuses_and_keeps_champion(promoted_gate, tmp_path) -> None:
    code, _, champion, _ = promoted_gate
    assert code == 0  # sanity: previous run installed a champion
    before = joblib.load(champion)["manifest"]["model_version"]

    impossible = QualityBar(max_median_latency=3, max_fp_rate=-1.0)
    code, report = run_gate(champion_path=champion, out_dir=tmp_path,
                            bar=impossible, **_TINY)
    assert code == 1
    assert "**REFUSED**" in report
    assert "FP rate" in report
    # the champion bundle is untouched
    assert joblib.load(champion)["manifest"]["model_version"] == before
    # the refused candidate is still archived
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "model.joblib").exists()
