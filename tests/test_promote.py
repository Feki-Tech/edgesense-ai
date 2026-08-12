"""Champion/challenger promotion gate: decision logic + end-to-end runs."""

from __future__ import annotations

import json

import joblib
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from ml.promote import (QualityBar, check_bar, check_shadow,  # noqa: E402
                        compare_to_champion, load_shadow_report,
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


# --- online evidence: the shadow report as a promotion criterion (§2.5) -----

VERSION = "20990101.000000+abcdef0"


def _shadow(n=5_000, agreement=0.99, version=VERSION, errors=0) -> dict:
    return {"shadow_version": version, "champion_version": "v1", "n": n,
            "agree": int(n * agreement), "agreement_rate": agreement,
            "champion_only": 0, "shadow_only": n - int(n * agreement),
            "errors": errors, "score_mae": 0.0002, "score_bias": -0.00004}


def test_shadow_accepts_strong_online_evidence() -> None:
    assert check_shadow(_shadow(), VERSION, QualityBar()) == []


def test_shadow_rejects_thin_evidence() -> None:
    failures = check_shadow(_shadow(n=12), VERSION, QualityBar())
    assert len(failures) == 1 and "too thin" in failures[0]


def test_shadow_rejects_low_agreement() -> None:
    failures = check_shadow(_shadow(agreement=0.80), VERSION, QualityBar())
    assert any("agreement" in f for f in failures)


def test_shadow_rejects_evidence_about_another_model() -> None:
    """Evidence earned by a different bundle says nothing about this one."""
    failures = check_shadow(_shadow(version="some-other-model"), VERSION, QualityBar())
    assert len(failures) == 1 and "--challenger" in failures[0]


def test_shadow_mismatch_suppresses_the_other_complaints() -> None:
    """A version mismatch makes every downstream number describe the wrong model."""
    failures = check_shadow(_shadow(version="other", n=3, agreement=0.1),
                            VERSION, QualityBar())
    assert len(failures) == 1


def test_shadow_rejects_scoring_errors() -> None:
    failures = check_shadow(_shadow(errors=4), VERSION, QualityBar())
    assert any("scoring error" in f for f in failures)


def test_shadow_report_reads_endpoint_and_bare_shapes(tmp_path) -> None:
    endpoint_shape = tmp_path / "endpoint.json"
    endpoint_shape.write_text(json.dumps({"active": True, "model": "x",
                                          "report": _shadow()}))
    bare_shape = tmp_path / "bare.json"
    bare_shape.write_text(json.dumps(_shadow()))

    assert load_shadow_report(str(endpoint_shape))["n"] == 5_000
    assert load_shadow_report(str(bare_shape))["n"] == 5_000


def test_shadow_report_reads_a_live_sidecar_url() -> None:
    """The documented workflow points the gate straight at a running node."""
    import http.server
    import threading

    payload = json.dumps({"active": True, "report": _shadow()}).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/shadow"
        assert load_shadow_report(url)["agreement_rate"] == 0.99
    finally:
        server.shutdown()


def test_shadow_report_surfaces_an_unreachable_sidecar() -> None:
    with pytest.raises(RuntimeError, match="could not fetch"):
        load_shadow_report("http://127.0.0.1:1/shadow")


def test_shadow_report_rejects_an_inactive_sidecar(tmp_path) -> None:
    path = tmp_path / "inactive.json"
    path.write_text(json.dumps({"active": False}))
    with pytest.raises(RuntimeError, match="no shadow is loaded"):
        load_shadow_report(str(path))


def test_report_renders_the_shadow_row() -> None:
    report = render_report(
        challenger_version=VERSION, champion_version="v1",
        challenger=_summary(), champion=_summary(), parity=(1e-6, 1.0),
        failures=[], promoted=True, bar=QualityBar(), shadow=_shadow())
    assert "shadow agreement on live traffic" in report
    assert "5,000 readings" in report


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


def test_gate_reuses_an_existing_challenger_instead_of_training(promoted_gate,
                                                                tmp_path) -> None:
    _, _, champion, out_dir = promoted_gate
    candidate = out_dir / "model.joblib"
    version = joblib.load(candidate)["manifest"]["model_version"]

    code, report = run_gate(champion_path=champion, challenger_path=candidate,
                            out_dir=tmp_path, bar=_TINY_BAR, **_TINY)
    assert code == 0
    assert version in report  # gated the bundle we handed it, not a fresh one


def test_gate_promotes_on_matching_shadow_evidence(promoted_gate, tmp_path) -> None:
    _, _, champion, out_dir = promoted_gate
    candidate = out_dir / "model.joblib"
    version = joblib.load(candidate)["manifest"]["model_version"]

    report_file = tmp_path / "shadow.json"
    report_file.write_text(json.dumps({"active": True,
                                       "report": _shadow(version=version)}))

    code, report = run_gate(champion_path=champion, challenger_path=candidate,
                            shadow_report=str(report_file), out_dir=tmp_path,
                            bar=_TINY_BAR, **_TINY)
    assert code == 0
    assert "shadow agreement on live traffic" in report
    # the evidence is recorded in the promoted model's manifest
    manifest = json.loads((champion.parent / "model.manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["metrics"]["shadow_agreement_rate"] == 0.99


def test_gate_refuses_when_shadow_evidence_is_weak(promoted_gate, tmp_path) -> None:
    """A model that clears every offline bar can still fail on live evidence."""
    _, _, champion, out_dir = promoted_gate
    candidate = out_dir / "model.joblib"
    version = joblib.load(candidate)["manifest"]["model_version"]
    before = joblib.load(champion)["manifest"]["model_version"]

    report_file = tmp_path / "shadow.json"
    report_file.write_text(json.dumps(_shadow(version=version, agreement=0.42)))

    code, report = run_gate(champion_path=champion, challenger_path=candidate,
                            shadow_report=str(report_file), out_dir=tmp_path,
                            bar=_TINY_BAR, **_TINY)
    assert code == 1
    assert "**REFUSED**" in report and "shadow agreement" in report
    assert joblib.load(champion)["manifest"]["model_version"] == before


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
