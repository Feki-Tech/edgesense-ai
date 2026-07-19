output "acr_login_server" {
  description = "Registry hostname to docker login / push images to."
  value       = azurerm_container_registry.this.login_server
}

output "acr_name" {
  description = "ACR name (use with: az acr build / az acr login --name)."
  value       = azurerm_container_registry.this.name
}

output "resource_group" {
  description = "Resource group holding everything."
  value       = azurerm_resource_group.this.name
}

output "dashboard_url" {
  description = "Public URL of the Streamlit dashboard (your demo link)."
  value       = "https://${azurerm_container_app.dashboard.ingress[0].fqdn}"
}

output "inference_internal_fqdn" {
  description = "Internal FQDN of the inference API (reachable inside the env)."
  value       = azurerm_container_app.inference.ingress[0].fqdn
}
