########################################################################
# Phase 2 — Azure ML workspace (MLflow-compatible tracking + registry)
#
# The workspace itself is FREE — you pay only for compute/endpoints you
# attach. Its dependencies (storage, key vault, app insights) cost cents
# at this scale. Safe to leave standing; the expensive part (managed
# online endpoint) is opt-in via CLI, not Terraform — see docs/AZUREML.md.
########################################################################

resource "azurerm_application_insights" "ml" {
  count               = var.enable_phase2_azureml ? 1 : 0
  name                = "${local.name}-ml-appi"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.this.id
  tags                = var.tags
}

resource "azurerm_key_vault" "ml" {
  count                    = var.enable_phase2_azureml ? 1 : 0
  name                     = "${local.name}kv${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.this.name
  location                 = azurerm_resource_group.this.location
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "standard"
  purge_protection_enabled = false
  tags                     = var.tags
}

resource "azurerm_storage_account" "ml" {
  count                    = var.enable_phase2_azureml ? 1 : 0
  name                     = "${local.name}mlsa${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.this.name
  location                 = azurerm_resource_group.this.location
  account_tier             = "Standard"
  account_replication_type = "LRS" # cheapest; fine for a demo workspace
  tags                     = var.tags
}

resource "azurerm_machine_learning_workspace" "this" {
  count                          = var.enable_phase2_azureml ? 1 : 0
  name                           = "${local.name}-mlw"
  resource_group_name            = azurerm_resource_group.this.name
  location                       = azurerm_resource_group.this.location
  application_insights_id        = azurerm_application_insights.ml[0].id
  key_vault_id                   = azurerm_key_vault.ml[0].id
  storage_account_id             = azurerm_storage_account.ml[0].id
  container_registry_id          = azurerm_container_registry.this.id # reuse Phase-1 ACR
  public_network_access_enabled  = true
  tags                           = var.tags

  identity {
    type = "SystemAssigned"
  }
}

output "aml_workspace_name" {
  description = "Azure ML workspace (also the MLflow tracking server). Null unless enable_phase2_azureml = true."
  value       = try(azurerm_machine_learning_workspace.this[0].name, null)
}

output "aml_mlflow_note" {
  value = var.enable_phase2_azureml ? "Get the MLflow tracking URI with: az ml workspace show -n ${azurerm_machine_learning_workspace.this[0].name} -g ${azurerm_resource_group.this.name} --query mlflow_tracking_uri -o tsv" : null
}
