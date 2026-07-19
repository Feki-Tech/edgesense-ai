# edgesense-azure

Azure Container Apps deployment for [EdgeSense AI](https://github.com/Feki-Tech/edgesense-ai) — **Phase 1** of the cloud MLOps roadmap.

Copy this folder into the `edgesense-ai` repo (or keep it beside it) and follow
[`docs/AZURE.md`](docs/AZURE.md).

```
edgesense-azure/
├── infra/                  # Terraform: ACR, Container Apps env, 5 apps
│   ├── versions.tf         # providers + (optional) remote backend
│   ├── variables.tf
│   ├── main.tf
│   ├── outputs.tf
│   └── terraform.tfvars.example
├── scripts/
│   └── build-and-push.sh   # az acr build for all four images
├── .github/workflows/
│   └── azure-deploy.yml     # OIDC CD: build in ACR + roll the apps
└── docs/
    └── AZURE.md            # step-by-step deploy + CD setup + caveats
```

**Cost:** Consumption plan + scale-to-zero → ~€0 when idle. `terraform destroy`
between demos. Comfortably inside a €200 credit.

## Roadmap

- **Phase 1** ✅ — Container Apps + ACR + Terraform + OIDC CD ([docs/AZURE.md](docs/AZURE.md))
- **Phase 2** ✅ — Azure ML workspace + MLflow registry + managed endpoint ([docs/AZUREML.md](docs/AZUREML.md)): `infra/azureml.tf`, `ml/register_model.py`, `ml/deploy_endpoint.sh`
- **Phase 3** ✅ — Key Vault secrets via managed identity + Azure Managed Grafana ([docs/PHASE3-4.md](docs/PHASE3-4.md)): `infra/phase3.tf`
- **Phase 4** ✅ — Drift-triggered continuous training: Grafana alert → `repository_dispatch` → serverless Azure ML job ([docs/PHASE3-4.md](docs/PHASE3-4.md)): `ml/retrain_job.yml`, `.github/workflows/retrain-on-drift.yml`
- **Later** — mTLS device identity, OTA model delivery, per-machine thresholds.
