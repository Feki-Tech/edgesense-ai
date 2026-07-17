"""Model manifest, versioning and backward compatibility."""

from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pytest

from ml.manifest import (data_sha256, manifest_path, render_model_card,
                         save_bundle)
from ml.scoring import score_sample
from ml.train import FEATURES, HIDDEN_LAYERS


def test_manifest_embedded_in_new_bundles(small_model_path) -> None:
    manifest = joblib.load(small_model_path)["manifest"]
    assert manifest["schema_version"] == 1
    assert re.fullmatch(r"\d{8}\.\d{6}\+[0-9a-f]{7}|\d{8}\.\d{6}\+nogit",
                        manifest["model_version"])
    assert manifest["created_at"].endswith("Z")
    assert manifest["backend"] == "sklearn"
    assert manifest["kind"] == "autoencoder"
    assert manifest["features"] == FEATURES

    training = manifest["training"]
    assert training["seed"] == 0 and training["epochs"] == 300
    assert training["architecture"]["layers"] == \
        [len(FEATURES), *HIDDEN_LAYERS, len(FEATURES)]
    assert training["fp_budget"] == 0.005
    assert training["n_train"] == 5_000 and training["n_cal"] == 3_000

    td = manifest["training_data"]
    assert td["generator"] == "synthetic-normal-v1"
    assert set(td["params"]) == set(FEATURES)
    assert re.fullmatch(r"[0-9a-f]{64}", td["sha256"])

    assert manifest["metrics"]["threshold"] == \
        pytest.approx(joblib.load(small_model_path)["threshold"])


def test_data_sha256_is_content_addressed() -> None:
    x = np.arange(12, dtype=float).reshape(4, 3)
    assert data_sha256(x) == data_sha256(x.copy())
    assert data_sha256(x) != data_sha256(x + 1)


def test_save_bundle_writes_sidecar_manifest_and_card(small_model_path, tmp_path) -> None:
    bundle = joblib.load(small_model_path)
    out = save_bundle(bundle, tmp_path / "model.joblib")

    sidecar = manifest_path(out)
    assert sidecar.name == "model.manifest.json"
    on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
    assert on_disk["model_version"] == bundle["manifest"]["model_version"]

    card = (tmp_path / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert bundle["manifest"]["model_version"] in card
    assert "# EdgeSense AI — model card" in card
    assert "3 → 16 → 2 → 16 → 3" in card


def test_model_card_renders_all_sections(small_model_path) -> None:
    manifest = joblib.load(small_model_path)["manifest"]
    card = render_model_card(manifest)
    for heading in ("## Architecture & training", "## Training data",
                    "## Metrics snapshot", "## Intended use & limits"):
        assert heading in card


def test_legacy_bundle_without_manifest_still_scores(legacy_model_path) -> None:
    """Pre-phase-1 bundles (no manifest key) keep loading and scoring."""
    bundle = joblib.load(legacy_model_path)
    assert "manifest" not in bundle

    _, is_anomaly, reason = score_sample(bundle, [0.8, 45.0, 12.0])
    assert not is_anomaly and reason is None
    _, is_anomaly, _ = score_sample(bundle, [4.5, 46.0, 14.0])
    assert is_anomaly


def test_save_bundle_without_manifest_writes_no_sidecar(legacy_model_path, tmp_path) -> None:
    bundle = joblib.load(legacy_model_path)
    out = save_bundle(bundle, tmp_path / "model.joblib")
    assert out.exists()
    assert not manifest_path(out).exists()
    assert not (tmp_path / "MODEL_CARD.md").exists()
