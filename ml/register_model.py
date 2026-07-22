#!/usr/bin/env python3
"""Register an EdgeSense model bundle in the Azure ML MLflow registry.

Bridges the repo's existing artifacts (model.joblib + model.manifest.json +
MODEL_CARD.md) into MLflow, so promotion becomes an auditable registry
operation instead of a file copy.

Flow:
  train  -> ml/train.py            (unchanged, produces the bundle)
  gate   -> ml/promote.py          (unchanged, champion/challenger)
  here   -> log bundle as an MLflow run + register a new model version,
            tagging it with the EdgeSense manifest version + metrics.

Usage:
  pip install "mlflow<3" azureml-mlflow azure-ai-ml azure-identity  # azureml-mlflow needs mlflow 2.x
  export MLFLOW_TRACKING_URI=$(az ml workspace show -n edgesense-mlw \
      -g edgesense-rg --query mlflow_tracking_uri -o tsv)
  python register_model.py --bundle ../edgesense-ai/ml/model \
      [--promote]   # also move the new version to the "champion" alias
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "edgesense-anomaly"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="Path to ml/model/ (model.joblib + model.manifest.json)")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--promote", action="store_true",
                    help="Tag the newly registered version as champion "
                         "(run ml/promote.py first — this script does NOT gate).")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    manifest_path = bundle / "model.manifest.json"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found — train a model first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    version_tag = manifest.get("model_version", "unknown")
    metrics = manifest.get("metrics", {})

    # Inside an Azure ML job there is already an active MLflow run (the job
    # itself) whose ID is fixed by the environment — starting a fresh run there
    # raises "active run ID does not match environment run ID". So reuse the
    # ambient run when present, and only start our own when run standalone
    # (e.g. the Phase-2 manual `python ml/register_model.py`).
    active = mlflow.active_run()
    if active is None:
        mlflow.set_experiment("edgesense-training")
        run_ctx = mlflow.start_run(run_name=f"register-{version_tag}")
    else:
        run_ctx = contextlib.nullcontext(active)

    with run_ctx as run:
        # Log the manifest's metric snapshot so the registry entry is self-describing.
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        mlflow.log_params({
            "edgesense_version": version_tag,
            "training_data_hash": manifest.get("training_data", {}).get("sha256", ""),
        })
        # Log the whole bundle (joblib + manifest + model card) as artifacts.
        mlflow.log_artifacts(str(bundle), artifact_path="model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)
        print(f"registered {args.model_name} v{mv.version} (edgesense {version_tag})")

    client = MlflowClient()
    client.set_model_version_tag(args.model_name, mv.version,
                                 "edgesense_version", version_tag)

    # Azure ML's MLflow registry doesn't implement the alias API (404), so
    # champion/challenger live in tags: a model-level 'champion_version'
    # pointer plus a per-version 'role' tag.
    if args.promote:
        client.set_model_version_tag(args.model_name, mv.version, "role", "champion")
        client.set_registered_model_tag(args.model_name, "champion_version", mv.version)
        print(f"champion_version -> v{mv.version}")
    else:
        client.set_model_version_tag(args.model_name, mv.version, "role", "challenger")
        print(f"v{mv.version} tagged challenger (run promote.py, then re-run with --promote)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
