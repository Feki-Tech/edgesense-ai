"""Export the trained pipeline to ONNX (ml/model/model.onnx).

Enables running inference without the Python sidecar, e.g. via onnxruntime
in the Go agent. The z-score guard is model-agnostic and must be applied by
the consumer (see ml/scoring.py); the scaler parameters needed for it are
stored alongside the ONNX file in model.onnx.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from skl2onnx import to_onnx

MODEL_DIR = Path(__file__).parent / "model"


def main() -> None:
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    pipeline = bundle["pipeline"]

    sample = np.array([[0.8, 45.0, 12.0]], dtype=np.float32)
    onx = to_onnx(pipeline, X=sample, target_opset={"": 21, "ai.onnx.ml": 3})

    out = MODEL_DIR / "model.onnx"
    out.write_bytes(onx.SerializeToString())

    scaler = pipeline.named_steps["scaler"]
    meta = {
        "features": bundle["features"],
        "z_guard": bundle.get("z_guard", 6.0),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    (MODEL_DIR / "model.onnx.json").write_text(json.dumps(meta, indent=2))

    import onnxruntime as ort
    sess = ort.InferenceSession(out.read_bytes(), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {sess.get_inputs()[0].name: sample})
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KiB)")
    print(f"outputs: {[o.name for o in sess.get_outputs()]}, nominal sample -> {outputs}")


if __name__ == "__main__":
    main()
