#!/bin/bash
set -euo pipefail

dnf update -y
dnf install -y docker
systemctl enable docker
systemctl start docker

cat >/opt/ml-ecommerce-chatbot.env <<EOF
ENV=prod
LLM_PROVIDER=${llm_provider}
AWS_REGION=${aws_region}
BEDROCK_MODEL_ID=${bedrock_model_id}
BEDROCK_ROLE_ARN=${bedrock_role_arn}
BEDROCK_OUTPUT_S3_URI=${bedrock_output_s3_uri}
PREDICTION_BACKEND=${prediction_backend}
SAGEMAKER_ENDPOINT_NAME=${sagemaker_endpoint}
DATABASE_URL=${database_url}
CLEARML_ENABLED=${clearml_enabled}
CLEARML_API_HOST=${clearml_api_host}
CLEARML_WEB_HOST=${clearml_web_host}
CLEARML_FILES_HOST=${clearml_files_host}
CLEARML_SERVING_BASE_URL=${clearml_serving_base_url}
CLEARML_SERVING_ENDPOINT=${clearml_serving_endpoint}
EOF

docker run -d \
  --restart unless-stopped \
  --name ml-ecommerce-chatbot \
  --env-file /opt/ml-ecommerce-chatbot.env \
  -p 8000:8000 \
  ${container_image}
