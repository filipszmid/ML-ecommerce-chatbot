output "chat_url" {
  value = "http://${aws_instance.chat.public_ip}:8000"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "sagemaker_endpoint_name" {
  value = aws_sagemaker_endpoint.category.name
}

output "predict_lambda_name" {
  value = aws_lambda_function.predict_proxy.function_name
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "postgres_endpoint" {
  value = aws_db_instance.chat.address
}
