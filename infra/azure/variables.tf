variable "project_name" {
  type    = string
  default = "ml-ecommerce-chatbot"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "location" {
  type    = string
  default = "westeurope"
}

variable "container_image" {
  type        = string
  description = "Container image for the deployed application service."
}

variable "azure_ml_model_id" {
  type        = string
  description = "Azure ML model asset id used by the online deployment."
}

variable "azure_ml_environment_id" {
  type        = string
  description = "Azure ML environment asset id used by the online deployment."
}

variable "azure_ml_scoring_uri" {
  type        = string
  default     = ""
  description = "Azure ML online endpoint scoring URI used by the Function proxy."
}

variable "azure_ml_api_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Azure ML online endpoint key used by the Function proxy."
}

variable "azure_openai_endpoint" {
  type        = string
  default     = ""
  description = "Existing Azure OpenAI-compatible endpoint."
}

variable "azure_openai_deployment" {
  type        = string
  default     = "ecommerce-chat"
  description = "Azure OpenAI deployment name."
}

variable "azure_openai_api_version" {
  type        = string
  default     = "2024-10-21"
  description = "Azure OpenAI API version used by chat and fine-tuning REST calls."
}

variable "azure_openai_fine_tune_model" {
  type        = string
  default     = "gpt-4.1-mini"
  description = "Azure OpenAI base model id used when creating fine-tuning jobs."
}

variable "azure_openai_api_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Azure OpenAI API key, normally injected from CI secrets."
}

variable "create_azure_openai" {
  type    = bool
  default = false
}

variable "llm_provider" {
  type        = string
  default     = "azure_openai"
  description = "LLM provider used by the deployed chat service."
}

variable "prediction_backend" {
  type        = string
  default     = "azure_ml"
  description = "Prediction backend used by the deployed API."
}

variable "clearml_enabled" {
  type        = bool
  default     = false
  description = "Enable ClearML tracking for Azure workloads."
}

variable "clearml_api_host" {
  type        = string
  default     = ""
  description = "ClearML API host reachable from Azure workloads."
}

variable "clearml_web_host" {
  type        = string
  default     = ""
  description = "ClearML Web UI host shown in tracked metadata."
}

variable "clearml_files_host" {
  type        = string
  default     = ""
  description = "ClearML file server host reachable from Azure workloads."
}

variable "clearml_serving_base_url" {
  type        = string
  default     = ""
  description = "External ClearML Serving base URL reachable from Azure workloads."
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
  default = "mlchatbot"
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
