variable "prefix" {
  description = "Short name prefix for all resources (lowercase, no spaces)."
  type        = string
  default     = "edgesense"
}

variable "location" {
  description = "Azure region. Frankfurt/Netherlands are close to Germany."
  type        = string
  default     = "germanywestcentral"
}

variable "image_tag" {
  description = "Container image tag to deploy (usually the git SHA or 'latest')."
  type        = string
  default     = "latest"
}

variable "tags" {
  description = "Tags applied to every resource (handy for cost tracking)."
  type        = map(string)
  default = {
    project = "edgesense-ai"
    owner   = "mohamed-feki"
    env     = "demo"
  }
}

variable "enable_phase2_azureml" {
  description = "Provision the Azure ML workspace (Phase 2: MLflow-compatible registry + its storage/Key Vault/App Insights). Off by default so a fresh apply stays Phase-1-only."
  type        = bool
  default     = false
}

variable "enable_phase3" {
  description = "Provision the Key Vault-backed broker secret + Azure Managed Grafana (Phase 3, ~EUR 8-10/month for Grafana). Requires enable_phase2_azureml = true (Phase 3 reuses the AML Key Vault)."
  type        = bool
  default     = false
}

variable "prometheus_auth_user" {
  description = "Basic-auth username for the Prometheus container app's public ingress (Grafana uses it as data source credentials)."
  type        = string
  default     = "grafana"
}

variable "prometheus_auth_bcrypt" {
  description = "Bcrypt hash of the Prometheus basic-auth password (precomputed, e.g. python -c \"import bcrypt; print(bcrypt.hashpw(pw, bcrypt.gensalt()).decode())\" — Terraform's own bcrypt() re-salts every run and would churn revisions). The plaintext lives in Key Vault (prometheus-basic-auth) and in the Grafana data source, not in Terraform."
  type        = string
  sensitive   = true
}
