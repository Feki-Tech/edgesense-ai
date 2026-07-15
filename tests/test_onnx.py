"""ONNX export parity: onnxruntime must reproduce the numpy scorer's verdicts."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from ml.export_onnx import build_onnx  # noqa: E402
from ml.scoring import reconstruction_errors  # noqa: E402
from ml.train import fault_data, normal_data  # noqa: E402


@pytest.fixture(scope="module")
def onnx_session_and_bundle(small_model_path):
    bundle = joblib.load(small_model_path)
    model = build_onnx(bundle)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess, bundle


def _run(sess, x: np.ndarray):
    outputs = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})
    named = {o.name: v for o, v in zip(sess.get_outputs(), outputs)}
    return np.ravel(named["label"]), np.ravel(named["scores"])


def test_onnx_matches_numpy_scorer(onnx_session_and_bundle) -> None:
    sess, bundle = onnx_session_and_bundle
    rng = np.random.default_rng(7)
    x = np.vstack([normal_data(500, rng), fault_data(500, rng)])

    onnx_labels, onnx_scores = _run(sess, x)
    ref_scores = reconstruction_errors(bundle, x)
    ref_labels = (ref_scores > bundle["threshold"]).astype(np.int64)

    # relative error: fault scores are large, healthy scores are tiny
    rel_mae = float(np.mean(np.abs(onnx_scores - ref_scores) / np.maximum(ref_scores, 1e-6)))
    agreement = float(np.mean(onnx_labels == ref_labels))
    assert rel_mae < 1e-3, f"score relative MAE too high: {rel_mae}"
    assert agreement > 0.99, f"label agreement too low: {agreement:.3%}"


def test_onnx_rejects_legacy_iforest_bundle(iforest_model_path) -> None:
    bundle = joblib.load(iforest_model_path)
    with pytest.raises(ValueError, match="autoencoder"):
        build_onnx(bundle)


def test_onnx_flags_extreme_bearing_fault(onnx_session_and_bundle) -> None:
    sess, bundle = onnx_session_and_bundle
    labels, scores = _run(sess, np.array([[4.5, 46.0, 14.0]]))
    assert labels[0] == 1
    assert scores[0] > bundle["threshold"]


def test_onnx_nominal_sample_is_normal(onnx_session_and_bundle) -> None:
    sess, bundle = onnx_session_and_bundle
    labels, scores = _run(sess, np.array([[0.8, 45.0, 12.0]]))
    assert labels[0] == 0
    assert 0 <= scores[0] <= bundle["threshold"]
