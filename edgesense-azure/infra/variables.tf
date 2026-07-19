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
