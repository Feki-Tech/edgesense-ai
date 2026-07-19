# EdgeSense AI on Azure — Phase 1 (Container Apps)

This directory takes the existing EdgeSense containers and runs them on
**Azure Container Apps**, provisioned with **Terraform** and shipped by a
**GitHub Actions** CD pipeline. It is the cheapest way to put "Azure +
Terraform + CD" on the CV while keeping inside the €200 credit.

## Architecture

```
                 Internet
                    │  https
             ┌──────▼───────┐   external ingress
             │  dashboard   │  (Streamlit, scale 0→1)
             └──────┬───────┘
   ┌────────────────┼─────────────── Container Apps Environment ──────────┐
   │                │ MQTT                                                 │
   │  ┌──────────┐  │        ┌────────────┐  https  ┌────────────┐        │
   │  │ simulator│──┼───────▶│   broker   │◀────────│  edge-agent│        │
   │  └──────────┘  │  TCP   │ (mosquitto)│  MQTT   └─────┬──────┘        │
   │                │  1883  └────────────┘               │ /score        │
   │                │                              ┌──────▼─────┐         │
   │                │                              │ inference  │ scale→0 │
   │                │                              │ (FastAPI)  │         │
   │                │                              └────────────┘         │
   └───── Log Analytics ◀── all app logs ──────────────────────────────────┘

   Images pulled from Azure Container Registry via a user-assigned
   managed identity (AcrPull) — no registry password stored anywhere.
```

## Prerequisites

- An Azure subscription with the €200 credit and the **free trial** or
  pay-as-you-go offer active.
- Azure CLI (`az`) and Terraform ≥ 1.6 locally, **or** just use Cloud Shell.
- The `edgesense-ai` repo checked out next to this folder (the build script
  points at its Dockerfiles).

## Deploy — first time

```bash
cd edgesense-azure/infra
az login                       # or run everything in Azure Cloud Shell
az account set --subscription <your-sub-id>

cp terraform.tfvars.example terraform.tfvars   # tweak if you like

# 1) Create ONLY the registry + environment first, so images have somewhere
#    to go before the apps try to pull them.
terraform init
terraform validate
terraform apply -target=azurerm_container_registry.this \
                -target=azurerm_container_app_environment.this \
                -target=azurerm_user_assigned_identity.apps \
                -target=azurerm_role_assignment.acr_pull

# 2) Build + push the four images into the new ACR.
ACR=$(terraform output -raw acr_name)
../scripts/build-and-push.sh "$ACR" latest ../../edgesense-ai

# 3) Now create the container apps (images exist, pulls will succeed).
terraform apply

terraform output dashboard_url   # open this in a browser
```

> Ordering matters only on the very first apply. After that, a normal
> `terraform apply` (or the CD pipeline) is enough.

## Tear down (do this between demo sessions to protect the credit)

```bash
terraform destroy
```

Scale-to-zero already means idle cost is tiny, but `destroy` takes it to €0.

## CD pipeline (GitHub Actions, OIDC)

`.github/workflows/azure-deploy.yml` builds images in ACR and rolls each app
on every push to `main`. Set it up once:

```bash
# App registration + service principal
az ad app create --display-name edgesense-cd
APP_ID=$(az ad app list --display-name edgesense-cd --query "[0].appId" -o tsv)
az ad sp create --id "$APP_ID"

# Federated credential so GitHub can log in with no secret
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "edgesense-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:Feki-Tech/edgesense-ai:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

# Grant it Contributor on the resource group
SUB=$(az account show --query id -o tsv)
az role assignment create --assignee "$APP_ID" --role Contributor \
  --scope "/subscriptions/$SUB/resourceGroups/edgesense-rg"
```

Then add these **repository variables** (Settings → Secrets and variables →
Actions → Variables): `AZURE_CLIENT_ID` (= `$APP_ID`), `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`, `ACR_NAME`, `RESOURCE_GROUP` (= `edgesense-rg`).

## Known caveats / things to iterate on

- **Broker auth**: Phase 1 runs an anonymous MQTT listener for simplicity.
  Move the demo credentials from `deploy/secure/` into **Azure Key Vault**
  and mount them as secrets in Phase 3.
- **Internal service DNS**: env vars wire apps together via each app's
  ingress FQDN (Terraform resolves these after apply). If the agent can't
  reach inference, confirm the inference app shows a healthy revision and
  that its internal ingress is enabled.
- **Prometheus/Grafana**: not deployed here — Container Apps already ships
  logs to Log Analytics. For metrics dashboards, add **Azure Managed
  Grafana** + the Prometheus scrape in Phase 3, or run them as two more
  container apps.
- **Mosquitto startup command**: the anonymous listener is written inline at
  container start. If you'd rather bake `deploy/mosquitto.conf`, build a tiny
  broker image and point the `broker` app at it.
- **CD vs Terraform drift**: the pipeline rolls images with `az containerapp
  update`, so the live image tag drifts from `var.image_tag` in state. Either
  ignore it (harmless) or add `lifecycle { ignore_changes = [template[0].container[0].image] }`
  to each app.
- **Agent buffer is ephemeral**: no volume is mounted, so buffered events die
  with the replica. For the real store-and-forward story, add an Azure Files
  volume to the agent app.
- **Dashboard scale-to-zero**: while scaled to zero it isn't subscribed to
  MQTT, so it only shows events from after it wakes. Set `min_replicas = 1`
  during demos.
- This is a **scaffold**: run `terraform plan` and adjust resource names,
  CPU/memory, and image paths to your exact Dockerfiles before applying.

## What this adds to your resume

> Deployed EdgeSense AI to **Azure Container Apps** (scale-to-zero
> Consumption plan), provisioned end-to-end with **Terraform** (ACR,
> managed-identity image pulls, Log Analytics), and automated releases with a
> **GitHub Actions → Azure** CD pipeline authenticating via **OIDC**.
