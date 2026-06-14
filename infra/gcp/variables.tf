variable "project_id" {
  type        = string
  description = "GCP project id."
}

variable "project_name" {
  type    = string
  default = "ml-ecommerce-chatbot"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "region" {
  type    = string
  default = "europe-west4"
}

variable "container_image" {
  type        = string
  description = "Container image for the deployed application service."
}

variable "model_display_name" {
  type    = string
  default = "ecommerce-product-category"
}

variable "model_artifact_uri" {
  type        = string
  description = "GCS URI with Vertex model artifacts."
}

variable "model_serving_image" {
  type        = string
  description = "Container image used for Vertex AI prediction."
}

variable "vertex_model_id" {
  type    = string
  default = "gemini-2.5-flash"
}

variable "llm_provider" {
  type        = string
  default     = "vertex"
  description = "LLM provider used by the deployed chat service."
}

variable "prediction_backend" {
  type        = string
  default     = "vertex"
  description = "Prediction backend used by the deployed API."
}

variable "clearml_enabled" {
  type        = bool
  default     = false
  description = "Enable ClearML tracking for GCP workloads."
}

variable "clearml_api_host" {
  type        = string
  default     = ""
  description = "ClearML API host reachable from GCP workloads."
}

variable "clearml_web_host" {
  type        = string
  default     = ""
  description = "ClearML Web UI host shown in tracked metadata."
}

variable "clearml_files_host" {
  type        = string
  default     = ""
  description = "ClearML file server host reachable from GCP workloads."
}

variable "clearml_serving_base_url" {
  type        = string
  default     = ""
  description = "External ClearML Serving base URL reachable from GCP workloads."
}

variable "clearml_serving_endpoint" {
  type        = string
  default     = "product-category"
  description = "ClearML Serving endpoint name used for product-category predictions."
}

variable "db_name" {
  type    = string
  default = "ml_ecommerce_chatbot"
}

variable "db_username" {
  type    = string
  default = "ml_chatbot"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "PostgreSQL password for the chat database."
}

variable "tags" {
  type    = map(string)
  default = {}
}
