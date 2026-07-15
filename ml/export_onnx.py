"""Export the trained autoencoder to ONNX (ml/model/model.onnx).

The graph is self-contained — the scaler, network weights and calibrated
alarm threshold are baked in as constants:

    input  X       float32 [n, 3]  raw features (vibration, temperature, current)
    output scores  float32 [n]     mean squared reconstruction error in scaled
                                   space (higher = more anomalous)
    output label   int64   [n]     1 = anomaly (scores > threshold), 0 = normal

Enables running inference without the Python sidecar, e.g. via onnxruntime in
the Go agent. The z-score guard is model-agnostic and must be applied by the
consumer (see ml/scoring.py); the scaler parameters needed for it are stored
alongside the ONNX file in model.onnx.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).parent / "model"


def build_onnx(bundle: dict):
    """Compose the scoring graph: scale -> autoencoder -> error -> threshold."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if bundle.get("kind") != "autoencoder":
        raise ValueError("ONNX export supports autoencoder bundles; retrain with "
                         "ml/train.py (the autoencoder is the default model)")
    if bundle["activation"] != "tanh":
        raise ValueError(f"unsupported activation: {bundle['activation']}")

    inits = [
        numpy_helper.from_array(np.asarray(bundle["scaler_mean"], np.float32), "mean"),
        numpy_helper.from_array(np.asarray(bundle["scaler_scale"], np.float32), "scale"),
        numpy_helper.from_array(np.asarray(bundle["threshold"], np.float32), "threshold"),
        numpy_helper.from_array(np.array([1], np.int64), "feature_axis"),
    ]
    nodes = [
        helper.make_node("Sub", ["X", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["z"]),
    ]

    h = "z"
    last = len(bundle["weights"]) - 1
    for i, (w, b) in enumerate(bundle["weights"]):
        inits += [numpy_helper.from_array(w.astype(np.float32), f"w{i}"),
                  numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        nodes.append(helper.make_node("MatMul", [h, f"w{i}"], [f"mm{i}"]))
        h = f"lin{i}"
        nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [h]))
        if i < last:
            nodes.append(helper.make_node("Tanh", [h], [f"act{i}"]))
            h = f"act{i}"

    nodes += [
        helper.make_node("Sub", ["z", h], ["residual"]),
        helper.make_node("Mul", ["residual", "residual"], ["squared"]),
        helper.make_node("ReduceMean", ["squared", "feature_axis"], ["scores"], keepdims=0),
        helper.make_node("Greater", ["scores", "threshold"], ["hit"]),
        helper.make_node("Cast", ["hit"], ["label"], to=TensorProto.INT64),
    ]

    n_features = len(bundle["features"])
    graph = helper.make_graph(
        nodes, "edgesense_autoencoder",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    return model


def main() -> None:
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    onx = build_onnx(bundle)

    out = MODEL_DIR / "model.onnx"
    out.write_bytes(onx.SerializeToString())

    meta = {
        "kind": bundle["kind"],
        "backend": bundle.get("backend"),
        "features": bundle["features"],
        "activation": bundle["activation"],
        "z_guard": bundle.get("z_guard", 6.0),
        "scaler_mean": np.asarray(bundle["scaler_mean"]).tolist(),
        "scaler_scale": np.asarray(bundle["scaler_scale"]).tolist(),
        "threshold": bundle["threshold"],
        "score_semantics": "mean squared reconstruction error in scaled space; "
                           "higher = more anomalous",
        "label_semantics": "1 = anomaly (scores > threshold), 0 = normal",
    }
    (MODEL_DIR / "model.onnx.json").write_text(json.dumps(meta, indent=2))

    import onnxruntime as ort
    sess = ort.InferenceSession(out.read_bytes(), providers=["CPUExecutionProvider"])
    sample = np.array([[0.8, 45.0, 12.0]], dtype=np.float32)
    outputs = sess.run(None, {sess.get_inputs()[0].name: sample})
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KiB)")
    print(f"outputs: {[o.name for o in sess.get_outputs()]}, nominal sample -> {outputs}")


if __name__ == "__main__":
    main()
