locals {
  name         = "${var.project_name}-${var.environment}"
  database_url = "postgresql+psycopg://${var.db_username}:${urlencode(var.db_password)}@${aws_db_instance.chat.address}:5432/${var.db_name}"
  tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

data "aws_caller_identity" "current" {}

resource "aws_vpc" "main" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = merge(local.tags, { Name = "${local.name}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.name}-igw" })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${local.name}-public-a" })
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.42.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${local.name}-public-b" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.name}-public-rt" })
}

resource "aws_route" "internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.main.id
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "chat" {
  name        = "${local.name}-chat-sg"
  description = "Chat service ingress"
  vpc_id      = aws_vpc.main.id
  tags        = local.tags

  ingress {
    description = "HTTP"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "postgres" {
  name        = "${local.name}-postgres-sg"
  description = "PostgreSQL ingress from application service"
  vpc_id      = aws_vpc.main.id
  tags        = local.tags

  ingress {
    description     = "PostgreSQL from application host"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.chat.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "chat" {
  name       = "${local.name}-postgres-subnets"
  subnet_ids = [aws_subnet.public.id, aws_subnet.public_b.id]
  tags       = local.tags
}

resource "aws_db_instance" "chat" {
  identifier              = "${local.name}-postgres"
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  db_name                 = var.db_name
  username                = var.db_username
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.chat.name
  vpc_security_group_ids  = [aws_security_group.postgres.id]
  publicly_accessible     = false
  backup_retention_period = 7
  deletion_protection     = false
  skip_final_snapshot     = true
  storage_encrypted       = true
  tags                    = local.tags
}

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${replace(local.name, "-", "")}-artifacts-"
  tags          = local.tags
}

resource "aws_ecr_repository" "api" {
  name                 = "${local.name}-api"
  image_tag_mutability = "MUTABLE"
  tags                 = local.tags
}

resource "aws_iam_role" "ec2" {
  name = "${local.name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role" "bedrock_customization" {
  name = "${local.name}-bedrock-customization-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "bedrock.amazonaws.com"
      }
      Action = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "aws:SourceAccount" = data.aws_caller_identity.current.account_id
        }
        ArnLike = {
          "aws:SourceArn" = "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-customization-job/*"
        }
      }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "bedrock_customization_s3" {
  name = "${local.name}-bedrock-customization-s3"
  role = aws_iam_role.bedrock_customization.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy" "ec2" {
  name = "${local.name}-ec2-policy"
  role = aws_iam_role.ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:CreateModelCustomizationJob",
          "bedrock:GetModelCustomizationJob",
          "bedrock:ListModelCustomizationJobs",
          "bedrock:StopModelCustomizationJob",
          "bedrock:GetCustomModel",
          "bedrock:ListCustomModels",
          "bedrock:GetFoundationModel",
          "bedrock:ListFoundationModels"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.bedrock_customization.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "bedrock.amazonaws.com"
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "chat" {
  name = "${local.name}-chat-profile"
  role = aws_iam_role.ec2.name
}

resource "aws_instance" "chat" {
  ami                         = data.aws_ami.amazon_linux.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.chat.id]
  iam_instance_profile        = aws_iam_instance_profile.chat.name
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/templates/user_data.sh", {
    container_image           = var.container_image
    aws_region                = var.aws_region
    llm_provider              = var.llm_provider
    bedrock_model_id          = var.bedrock_model_id
    bedrock_role_arn          = aws_iam_role.bedrock_customization.arn
    bedrock_output_s3_uri     = "s3://${aws_s3_bucket.artifacts.bucket}/bedrock-finetuning-output/"
    prediction_backend        = var.prediction_backend
    sagemaker_endpoint        = aws_sagemaker_endpoint.category.name
    database_url              = local.database_url
    clearml_enabled           = var.clearml_enabled
    clearml_api_host          = var.clearml_api_host
    clearml_web_host          = var.clearml_web_host
    clearml_files_host        = var.clearml_files_host
    clearml_serving_base_url  = var.clearml_serving_base_url
    clearml_serving_endpoint  = var.clearml_serving_endpoint
  })

  tags = merge(local.tags, { Name = "${local.name}-chat" })
}

resource "aws_iam_role" "sagemaker" {
  name = "${local.name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_ecr" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy" "sagemaker_artifacts" {
  name = "${local.name}-sagemaker-artifacts"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
    }]
  })
}

resource "aws_sagemaker_model" "category" {
  name               = "${local.name}-category-model"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = var.model_image
    model_data_url = var.model_data_url
    environment = {
      MODEL_ARTIFACT_DIR = "/opt/ml/model"
    }
  }

  tags = local.tags
}

resource "aws_sagemaker_endpoint_configuration" "category" {
  name = "${local.name}-category-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.category.name
    initial_instance_count = 1
    instance_type          = var.sagemaker_instance_type
  }

  tags = local.tags
}

resource "aws_sagemaker_endpoint" "category" {
  name                 = "${local.name}-category"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.category.name
  tags                 = local.tags
}

data "archive_file" "lambda_proxy" {
  type        = "zip"
  source_file = "${path.module}/lambda/predict_proxy.py"
  output_path = "${path.module}/lambda/predict_proxy.zip"
}

resource "aws_iam_role" "lambda" {
  name = "${local.name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_predict" {
  name = "${local.name}-lambda-predict"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "sagemaker:InvokeEndpoint"
      ]
      Resource = aws_sagemaker_endpoint.category.arn
    }]
  })
}

resource "aws_lambda_function" "predict_proxy" {
  function_name    = "${local.name}-predict-proxy"
  role             = aws_iam_role.lambda.arn
  handler          = "predict_proxy.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.lambda_proxy.output_path
  source_code_hash = data.archive_file.lambda_proxy.output_base64sha256
  timeout          = 30

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.category.name
    }
  }

  tags = local.tags
}
