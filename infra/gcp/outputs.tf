output "chat_url" {
  value = google_cloud_run_v2_service.chat.uri
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.main.name
}

output "artifact_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "vertex_endpoint" {
  value = google_vertex_ai_endpoint.category.name
}

output "predict_function_name" {
  value = google_cloudfunctions2_function.predict_proxy.name
}

output "postgres_connection_name" {
  value = google_sql_database_instance.chat.connection_name
}
