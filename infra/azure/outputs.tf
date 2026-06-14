output "chat_url" {
  value = "https://${azurerm_linux_web_app.chat.default_hostname}"
}

output "container_registry" {
  value = azurerm_container_registry.main.login_server
}

output "machine_learning_workspace" {
  value = azurerm_machine_learning_workspace.main.name
}

output "online_endpoint_name" {
  value = azapi_resource.online_endpoint.name
}

output "function_app_name" {
  value = azurerm_linux_function_app.predict_proxy.name
}

output "postgres_fqdn" {
  value = azurerm_postgresql_flexible_server.chat.fqdn
}
