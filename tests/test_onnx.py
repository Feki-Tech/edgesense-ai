"""ONNX export parity: onnxruntime must reproduce sklearn's verdicts."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

skl2onnx = pytest.importorskip("skl2onnx")
ort = pytest.importorskip("onnxruntime")

from ml.train import fault_data, normal_data  # noqa: E402


@pytest.fixture(scope="module")
def onnx_session_and_pipeline(small_model_path):
    pipeline = joblib.load(small_model_path)["pipeline"]
    sample = np.array([[0.8, 45.0, 12.0]], dtype=np.float32)
    try:
        onx = skl2onnx.to_onnx(pipeline, X=sample, target_opset={"": 21, "ai.onnx.ml": 3})
    except Exception as exc:  # pragma: no cover - depends on skl2onnx/sklearn combo
        pytest.skip(f"skl2onnx cannot convert this sklearn version: {exc}")
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess, pipeline


def _run(sess, x: np.ndarray):
    outputs = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})
    named = {o.name: v for o, v in zip(sess.get_outputs(), outputs)}
    return np.ravel(named["label"]), np.ravel(named["scores"])


def test_onnx_matches_sklearn(onnx_session_and_pipeline) -> None:
    sess, pipeline = onnx_session_and_pipeline
    rng = np.random.default_rng(7)
    x = np.vstack([normal_data(500, rng), fault_data(500, rng)])

    onnx_labels, onnx_scores = _run(sess, x)
    skl_labels = pipeline.predict(x)
    skl_scores = pipeline.decision_function(x)

    mae = float(np.mean(np.abs(onnx_scores - skl_scores)))
    agreement = float(np.mean(onnx_labels == skl_labels))
    assert mae < 1e-3, f"score MAE too high: {mae}"
    assert agreement > 0.99, f"label agreement too low: {agreement:.3%}"


def test_onnx_flags_extreme_bearing_fault(onnx_session_and_pipeline) -> None:
    sess, _ = onnx_session_and_pipeline
    labels, scores = _run(sess, np.array([[4.5, 46.0, 14.0]]))
    assert labels[0] == -1
    assert scores[0] < 0
