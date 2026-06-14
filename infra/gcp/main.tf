locals {
  name         = "${var.project_name}-${var.environment}"
  database_url = "postgresql+psycopg://${var.db_username}:${urlencode(var.db_password)}@/${var.db_name}?host=/cloudsql/${google_sql_database_instance.chat.connection_name}"
  labels = merge(
    {
      project     = replace(var.project_name, "-", "_")
      environment = var.environment
      managed_by  = "terraform"
    },
    var.tags
  )
}

resource "google_project_service" "services" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
  ])

  project = var.project_id
  service = each.value
}

resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = replace(local.name, "-", "")
  description   = "Container images for ${local.name}"
  format        = "DOCKER"
  labels        = local.labels

  depends_on = [google_project_service.services]
}

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-${local.name}-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  labels                      = local.labels

  depends_on = [google_project_service.services]
}

resource "google_service_account" "chat" {
  account_id   = substr(replace("${local.name}-chat", "-", ""), 0, 28)
  display_name = "Ecommerce chatbot service account"
}

resource "google_project_iam_member" "chat_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.chat.email}"
}

resource "google_project_iam_member" "chat_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.chat.email}"
}

resource "google_project_iam_member" "chat_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.chat.email}"
}

resource "google_storage_bucket_iam_member" "chat_artifacts_object_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.chat.email}"
}

resource "google_sql_database_instance" "chat" {
  name                = replace("${local.name}-postgres", "-", "")
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = false

  settings {
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    disk_size         = 10
    disk_type         = "PD_SSD"

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      ipv4_enabled = false
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_sql_database" "chat" {
  name     = var.db_name
  instance = google_sql_database_instance.chat.name
}

resource "google_sql_user" "chat" {
  name     = var.db_username
  instance = google_sql_database_instance.chat.name
  password = var.db_password
}

resource "google_cloud_run_v2_service" "chat" {
  name     = local.name
  location = var.region
  labels   = local.labels

  template {
    service_account = google_service_account.chat.email

    volumes {
      name = "cloudsql"

      cloud_sql_instance {
        instances = [google_sql_database_instance.chat.connection_name]
      }
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8000
      }

      env {
        name  = "ENV"
        value = "prod"
      }

      env {
        name  = "LLM_PROVIDER"
        value = var.llm_provider
      }

      env {
        name  = "PREDICTION_BACKEND"
        value = var.prediction_backend
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCP_LOCATION"
        value = var.region
      }

      env {
        name  = "VERTEX_MODEL_ID"
        value = var.vertex_model_id
      }

      env {
        name  = "VERTEX_ENDPOINT_ID"
        value = google_vertex_ai_endpoint.category.name
      }

      env {
        name  = "DATABASE_URL"
        value = local.database_url
      }

      env {
        name  = "CLEARML_ENABLED"
        value = var.clearml_enabled ? "true" : "false"
      }

      env {
        name  = "CLEARML_API_HOST"
        value = var.clearml_api_host
      }

      env {
        name  = "CLEARML_WEB_HOST"
        value = var.clearml_web_host
      }

      env {
        name  = "CLEARML_FILES_HOST"
        value = var.clearml_files_host
      }

      env {
        name  = "CLEARML_SERVING_BASE_URL"
        value = var.clearml_serving_base_url
      }

      env {
        name  = "CLEARML_SERVING_ENDPOINT"
        value = var.clearml_serving_endpoint
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
  }

  depends_on = [
    google_project_iam_member.chat_cloudsql_client,
    google_storage_bucket_iam_member.chat_artifacts_object_admin,
    google_project_service.services,
    google_sql_database.chat,
    google_sql_user.chat,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  location = google_cloud_run_v2_service.chat.location
  name     = google_cloud_run_v2_service.chat.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_vertex_ai_model" "category" {
  display_name = var.model_display_name
  region       = var.region
  labels       = local.labels

  artifact_uri = var.model_artifact_uri

  container_spec {
    image_uri = var.model_serving_image
  }

  depends_on = [google_project_service.services]
}

resource "google_vertex_ai_endpoint" "category" {
  name         = "${local.name}-category"
  display_name = "${local.name}-category"
  location     = var.region
  labels       = local.labels

  depends_on = [google_project_service.services]
}

resource "google_vertex_ai_endpoint_deployed_model" "category" {
  endpoint = google_vertex_ai_endpoint.category.id
  model    = google_vertex_ai_model.category.id

  deployed_model {
    display_name = "blue"

    dedicated_resources {
      machine_spec {
        machine_type = "n1-standard-2"
      }
      min_replica_count = 1
      max_replica_count = 1
    }
  }
}

data "archive_file" "function_source" {
  type        = "zip"
  source_dir  = "${path.module}/function"
  output_path = "${path.module}/function/predict_proxy.zip"
}

resource "google_storage_bucket_object" "function_source" {
  name   = "functions/predict_proxy.zip"
  bucket = google_storage_bucket.artifacts.name
  source = data.archive_file.function_source.output_path
}

resource "google_cloudfunctions2_function" "predict_proxy" {
  name     = "${local.name}-predict-proxy"
  location = var.region
  labels   = local.labels

  build_config {
    runtime     = "python312"
    entry_point = "handler"
    source {
      storage_source {
        bucket = google_storage_bucket.artifacts.name
        object = google_storage_bucket_object.function_source.name
      }
    }
  }

  service_config {
    available_memory      = "256M"
    timeout_seconds       = 30
    service_account_email = google_service_account.chat.email
    environment_variables = {
      GCP_PROJECT_ID           = var.project_id
      GCP_LOCATION             = var.region
      VERTEX_ENDPOINT_ID       = google_vertex_ai_endpoint.category.name
      CLEARML_ENABLED          = var.clearml_enabled ? "true" : "false"
      CLEARML_API_HOST         = var.clearml_api_host
      CLEARML_WEB_HOST         = var.clearml_web_host
      CLEARML_FILES_HOST       = var.clearml_files_host
      CLEARML_SERVING_BASE_URL = var.clearml_serving_base_url
      CLEARML_SERVING_ENDPOINT = var.clearml_serving_endpoint
    }
  }

  depends_on = [google_project_service.services]
}
