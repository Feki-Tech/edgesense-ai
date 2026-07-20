# Phase 2 — Azure ML: MLflow registry + managed serving

Phase 1 got the containers to the cloud. Phase 2 replaces the hand-rolled
model handoff with an **MLflow model registry** hosted by an **Azure ML
workspace**, and adds an optional **managed online endpoint** for the
"deployed a model on Azure ML" resume line.

## What's in this phase

| File | Purpose |
|---|---|
| `infra/azureml.tf` | Workspace + its required storage/Key Vault/App Insights (reuses Phase-1 ACR + Log Analytics) |
| `ml/register_model.py` (in the repo's `ml/`, next to `train.py`) | Logs the trained bundle to MLflow and registers a version, tagged with your manifest version + metrics |
| `ml/deploy_endpoint.sh` | Creates the managed online endpoint + deployment (demo, then delete) |
| `ml/deployment.yml`, `ml/score.py`, `ml/environment.yml` | The deployment spec and scoring entry |

## Cost model — read this first

- **Workspace, registry, tracking: free.** Storage/KV/App Insights: cents.
  Safe to leave standing.
- **Managed online endpoint: NOT free** — the VM bills 24/7 while it exists
  (Standard_DS2_v2 ≈ €90/month). Create it, demo it, screenshot it,
  **delete it same day**. Your Container Apps sidecar stays the cheap
  always-on serving path.

## Workflow

```bash
# 0) provision (from infra/)
# In terraform.tfvars, set: enable_phase2_azureml = true
terraform apply          # adds the workspace to the existing stack

# 1) point MLflow at the workspace
pip install "mlflow<3" azureml-mlflow azure-ai-ml azure-identity  # azureml-mlflow needs mlflow 2.x
export MLFLOW_TRACKING_URI=$(az ml workspace show -n edgesense-mlw \
    -g edgesense-rg --query mlflow_tracking_uri -o tsv)

# 2) train + gate exactly as before (in the edgesense-ai repo)
make train
make promote             # champion/challenger gate — unchanged

# 3) register the winning bundle (from the repo root)
python ml/register_model.py --bundle ml/model --promote

# 4) (optional, costs money) managed endpoint demo
./ml/deploy_endpoint.sh edgesense-mlw edgesense-rg
# ... demo, screenshot ...
az ml online-endpoint delete -n edgesense-anomaly-ep -w edgesense-mlw -g edgesense-rg --yes
```

## How this integrates with the existing MLOps loop

- `promote.py` stays the **quality gate**; MLflow becomes the **system of
  record**. Gate passes → `--promote` flips the `champion` alias. Gate not
  yet run → version lands as `challenger`.
- Every registered version carries the EdgeSense manifest version
  (`{YYYYMMDD.HHMMSS}+{git7}`) and the manifest's metric snapshot, so the
  registry UI answers "what's deployed and how good is it" at a glance.
- The Container Apps inference sidecar can later pull `@champion` from the
  registry at startup instead of baking the model at build time — that's the
  natural Phase 2.5 refactor, and it makes `POST /reload` a true
  registry-driven rollout.

## Caveats

- `score.py` re-implements the hybrid scoring rule; **verify the bundle key
  names** against `ml/scoring.py` (marked in the code). Better long-term:
  package `ml/` and import it in the scoring script.
- `deployment.yml` references `azureml:edgesense-anomaly@champion` — register
  with `--promote` at least once before deploying.
- The workspace needs your Azure AD identity to have `Contributor` on the
  resource group (you have it as subscription owner).

## Resume line this earns

> Managed the model lifecycle with **MLflow on Azure ML**: registry-backed
> champion/challenger promotion with aliased rollouts, and served the
> champion via an **Azure ML managed online endpoint**.
