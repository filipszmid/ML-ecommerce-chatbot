locals {
  name         = "${var.project_name}-${var.environment}"
  compact_name = substr(replace(local.name, "-", ""), 0, 20)
  database_url = "postgresql+psycopg://${var.db_username}:${urlencode(var.db_password)}@${azurerm_postgresql_flexible_server.chat.fqdn}:5432/${var.db_name}?sslmode=require"
  tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.52.0.0/16"]
  tags                = local.tags
}

resource "azurerm_subnet" "apps" {
  name                 = "snet-apps"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.52.1.0/24"]

  delegation {
    name = "web-apps"

    service_delegation {
      name = "Microsoft.Web/serverFarms"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/action",
      ]
    }
  }
}

resource "azurerm_subnet" "postgres" {
  name                 = "snet-postgres"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.52.2.0/24"]

  delegation {
    name = "postgres"

    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
      ]
    }
  }
}

resource "azurerm_container_registry" "main" {
  name                = "${local.compact_name}acr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = local.tags
}

resource "azurerm_storage_account" "main" {
  name                     = "${local.compact_name}st"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = local.tags
}

resource "azurerm_key_vault" "main" {
  name                       = "${local.compact_name}-kv"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  purge_protection_enabled   = false
  soft_delete_retention_days = 7
  tags                       = local.tags
}

data "azurerm_client_config" "current" {}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  tags                = local.tags
}

resource "azurerm_machine_learning_workspace" "main" {
  name                    = "mlw-${local.name}"
  location                = azurerm_resource_group.main.location
  resource_group_name     = azurerm_resource_group.main.name
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

resource "azurerm_private_dns_zone" "postgres" {
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${local.name}-postgres-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  tags                  = local.tags
}

resource "azurerm_postgresql_flexible_server" "chat" {
  name                   = "${local.name}-pg"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "16"
  delegated_subnet_id    = azurerm_subnet.postgres.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = var.db_username
  administrator_password = var.db_password
  zone                   = "1"
  storage_mb             = 32768
  sku_name               = "B_Standard_B1ms"
  tags                   = local.tags

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "chat" {
  name      = var.db_name
  server_id = azurerm_postgresql_flexible_server.chat.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

resource "azapi_resource" "online_endpoint" {
  type      = "Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2024-04-01"
  name      = "${local.name}-category"
  parent_id = azurerm_machine_learning_workspace.main.id
  location  = azurerm_resource_group.main.location

  body = {
    properties = {
      authMode = "Key"
    }
    identity = {
      type = "SystemAssigned"
    }
  }

  tags = local.tags
}

resource "azapi_resource" "online_deployment" {
  type      = "Microsoft.MachineLearningServices/workspaces/onlineEndpoints/deployments@2024-04-01"
  name      = "blue"
  parent_id = azapi_resource.online_endpoint.id
  location  = azurerm_resource_group.main.location

  body = {
    sku = {
      name     = "Default"
      capacity = 1
    }
    properties = {
      model         = var.azure_ml_model_id
      environmentId = var.azure_ml_environment_id
      instanceType  = "Standard_DS3_v2"
      instanceCount = 1
      requestSettings = {
        requestTimeout = "PT30S"
      }
    }
  }

  tags = local.tags
}

resource "azurerm_service_plan" "main" {
  name                = "plan-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  os_type             = "Linux"
  sku_name            = "B1"
  tags                = local.tags
}

resource "azurerm_linux_web_app" "chat" {
  name                      = "app-${local.name}"
  location                  = azurerm_resource_group.main.location
  resource_group_name       = azurerm_resource_group.main.name
  service_plan_id           = azurerm_service_plan.main.id
  https_only                = true
  virtual_network_subnet_id = azurerm_subnet.apps.id
  tags                      = local.tags

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      docker_image_name   = var.container_image
      docker_registry_url = "https://${azurerm_container_registry.main.login_server}"
    }
  }

  app_settings = {
    ENV                            = "prod"
    LLM_PROVIDER                   = var.llm_provider
    PREDICTION_BACKEND             = var.prediction_backend
    AZURE_OPENAI_ENDPOINT          = var.azure_openai_endpoint
    AZURE_OPENAI_DEPLOYMENT        = var.azure_openai_deployment
    AZURE_OPENAI_API_VERSION       = var.azure_openai_api_version
    AZURE_OPENAI_FINE_TUNE_MODEL   = var.azure_openai_fine_tune_model
    AZURE_OPENAI_API_KEY           = var.azure_openai_api_key
    AZURE_ML_ENDPOINT_NAME         = azapi_resource.online_endpoint.name
    DATABASE_URL                   = local.database_url
    CLEARML_ENABLED                = var.clearml_enabled ? "true" : "false"
    CLEARML_API_HOST               = var.clearml_api_host
    CLEARML_WEB_HOST               = var.clearml_web_host
    CLEARML_FILES_HOST             = var.clearml_files_host
    CLEARML_SERVING_BASE_URL       = var.clearml_serving_base_url
    CLEARML_SERVING_ENDPOINT       = var.clearml_serving_endpoint
  }
}

resource "azurerm_linux_function_app" "predict_proxy" {
  name                       = "func-${local.name}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  service_plan_id            = azurerm_service_plan.main.id
  storage_account_name       = azurerm_storage_account.main.name
  storage_account_access_key = azurerm_storage_account.main.primary_access_key
  https_only                 = true
  tags                       = local.tags

  site_config {
    application_stack {
      python_version = "3.12"
    }
  }

  app_settings = {
    FUNCTIONS_WORKER_RUNTIME  = "python"
    AZURE_ML_ENDPOINT_NAME    = azapi_resource.online_endpoint.name
    AZURE_ML_SCORING_URI      = var.azure_ml_scoring_uri
    AZURE_ML_API_KEY          = var.azure_ml_api_key
    DATABASE_URL              = local.database_url
    CLEARML_ENABLED           = var.clearml_enabled ? "true" : "false"
    CLEARML_API_HOST          = var.clearml_api_host
    CLEARML_WEB_HOST          = var.clearml_web_host
    CLEARML_FILES_HOST        = var.clearml_files_host
    CLEARML_SERVING_BASE_URL       = var.clearml_serving_base_url
    CLEARML_SERVING_ENDPOINT       = var.clearml_serving_endpoint
  }
}

resource "azurerm_cognitive_account" "openai" {
  count                 = var.create_azure_openai ? 1 : 0
  name                  = "oai-${local.name}"
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  kind                  = "OpenAI"
  sku_name              = "S0"
  custom_subdomain_name = "oai-${local.compact_name}"
  tags                  = local.tags
}
