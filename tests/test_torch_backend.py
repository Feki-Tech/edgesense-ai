"""PyTorch training backend: same bundle format and quality (optional dep)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from ml.scoring import reconstruction_errors, score_sample  # noqa: E402
from ml.train import (FEATURES, HIDDEN_LAYERS, fault_data,  # noqa: E402
                      normal_data, train_autoencoder)


@pytest.fixture(scope="module")
def torch_bundle() -> dict:
    return train_autoencoder("torch", seed=0, n_train=4_000, n_cal=2_000, epochs=120)


def test_bundle_format_matches_sklearn_backend(torch_bundle) -> None:
    assert torch_bundle["kind"] == "autoencoder"
    assert torch_bundle["backend"] == "torch"
    assert torch_bundle["features"] == FEATURES
    assert torch_bundle["threshold"] > 0
    dims = (len(FEATURES), *HIDDEN_LAYERS, len(FEATURES))
    for i, (w, b) in enumerate(torch_bundle["weights"]):
        assert w.shape == (dims[i], dims[i + 1])
        assert b.shape == (dims[i + 1],)
        assert w.dtype == np.float64 and b.dtype == np.float64


def test_torch_backend_meets_quality_gates(torch_bundle) -> None:
    rng = np.random.default_rng(1)
    thr = torch_bundle["threshold"]
    fp_rate = float(np.mean(reconstruction_errors(torch_bundle, normal_data(2_000, rng)) > thr))
    detection = float(np.mean(reconstruction_errors(torch_bundle, fault_data(2_000, rng)) > thr))
    assert fp_rate < 0.03, f"too many false positives: {fp_rate:.3%}"
    assert detection > 0.40, f"fault detection too weak: {detection:.3%}"


def test_score_sample_dispatch(torch_bundle) -> None:
    _, is_anomaly, reason = score_sample(torch_bundle, [0.8, 45.0, 12.0])
    assert not is_anomaly and reason is None
    _, is_anomaly, reason = score_sample(torch_bundle, [4.5, 46.0, 14.0])
    assert is_anomaly and reason in ("model", "limit", "model+limit")


def test_onnx_export_works_for_torch_bundle(torch_bundle) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from ml.export_onnx import build_onnx

    sess = ort.InferenceSession(build_onnx(torch_bundle).SerializeToString(),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(3)
    x = np.vstack([normal_data(200, rng), fault_data(200, rng)]).astype(np.float32)
    labels, scores = (np.ravel(v) for v in sess.run(None, {"X": x}))

    ref_scores = reconstruction_errors(torch_bundle, x)
    ref_labels = (ref_scores > torch_bundle["threshold"]).astype(np.int64)
    rel_mae = float(np.mean(np.abs(scores - ref_scores) / np.maximum(ref_scores, 1e-6)))
    assert rel_mae < 1e-3
    assert float(np.mean(labels == ref_labels)) > 0.99
