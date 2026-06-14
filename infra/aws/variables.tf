variable "project_name" {
  type    = string
  default = "ml-ecommerce-chatbot"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "container_image" {
  type        = string
  description = "Container image for the deployed application service."
}

variable "model_image" {
  type        = string
  description = "Inference container image for SageMaker."
}

variable "model_data_url" {
  type        = string
  description = "S3 URL to model.tar.gz with model.joblib and metadata."
}

variable "instance_type" {
  type    = string
  default = "t3.small"
}

variable "sagemaker_instance_type" {
  type    = string
  default = "ml.m5.large"
}

variable "bedrock_model_id" {
  type    = string
  default = "meta.llama3-1-8b-instruct-v1:0"
}

variable "llm_provider" {
  type        = string
  default     = "bedrock"
  description = "LLM provider used by the deployed chat service."
}

variable "prediction_backend" {
  type        = string
  default     = "sagemaker"
  description = "Prediction backend used by the deployed API."
}

variable "clearml_enabled" {
  type        = bool
  default     = false
  description = "Enable ClearML tracking for cloud-hosted app or training containers."
}

variable "clearml_api_host" {
  type        = string
  default     = ""
  description = "ClearML API host reachable from AWS workloads."
}

variable "clearml_web_host" {
  type        = string
  default     = ""
  description = "ClearML Web UI host shown in tracked metadata."
}

variable "clearml_files_host" {
  type        = string
  default     = ""
  description = "ClearML file server host reachable from AWS workloads."
}

variable "clearml_serving_base_url" {
  type        = string
  default     = ""
  description = "External ClearML Serving base URL reachable from AWS workloads."
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

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "allowed_cidr_blocks" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

variable "tags" {
  type    = map(string)
  default = {}
}
