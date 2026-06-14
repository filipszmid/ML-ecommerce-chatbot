"""Project-level configuration and non-secret defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if os.getenv("MASTER_CONFIG_SKIP_DOTENV", "false").lower() != "true":
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass


def _optional_int_env(name: str, default: str = "") -> int | None:
    value = os.getenv(name, default)
    return int(value) if value else None


def _csv_env(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


ENV = os.getenv("ENV", "dev")
PROJECT_NAME = os.getenv("PROJECT_NAME", "ml-ecommerce-chatbot")

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
CLASSICAL_RUNS_DIR = RUNS_DIR / "classical_ml"
LLM_RUNS_DIR = RUNS_DIR / "llm_finetuning"
EVALS_DIR = DATA_DIR / "evals"
PICTURES_DIR = DATA_DIR / "pictures"
CHAT_DATA_DIR = DATA_DIR / "chat"
DEFAULT_DATASET_PATH = DATA_DIR / "customer_purchase_data.csv"

TARGET_COLUMN = "ProductCategory"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "AnnualIncome",
    "NumberOfPurchases",
    "TimeSpentOnWebsite",
    "LoyaltyProgram",
    "DiscountsAvailed",
    "PurchaseStatus",
]

PRODUCT_CATEGORY_LABELS = {
    0: "Electronics",
    1: "Clothing",
    2: "Home Goods",
    3: "Beauty",
    4: "Sports",
}

DEFAULT_RANDOM_STATE = int(os.getenv("ML_RANDOM_STATE", "42"))
DEFAULT_TEST_SIZE = float(os.getenv("ML_TEST_SIZE", "0.2"))
DEFAULT_MAX_EVALS = int(os.getenv("ML_MAX_EVALS", "20"))
DEFAULT_CV_FOLDS = int(os.getenv("ML_CV_FOLDS", "3"))
DEFAULT_USE_SMOTE = os.getenv("ML_USE_SMOTE", "true").lower() == "true"
DEFAULT_SMOTE_K_NEIGHBORS = int(os.getenv("ML_SMOTE_K_NEIGHBORS", "5"))

MODEL_ARTIFACT_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", str(CLASSICAL_RUNS_DIR)))
MODEL_SELECTION_METRIC = os.getenv("MODEL_SELECTION_METRIC", "f1_macro")

POSTGRES_USER = os.getenv("POSTGRES_USER", "ml_chatbot")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ml_ecommerce_chatbot")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "ml_chatbot")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")
ADK_PORT = int(os.getenv("ADK_PORT", "8001"))
ADK_WEB_URL = os.getenv("ADK_WEB_URL", f"http://localhost:{ADK_PORT}")
ADK_AGENTS_DIR = os.getenv("ADK_AGENTS_DIR", "interface/chat")
CORS_ALLOW_ORIGINS = _csv_env(
    "CORS_ALLOW_ORIGINS",
    ",".join(
        [
            ADK_WEB_URL,
            f"http://localhost:{ADK_PORT}",
            f"http://127.0.0.1:{ADK_PORT}",
        ]
    ),
)
ADK_SESSION_SERVICE_URI = os.getenv(
    "ADK_SESSION_SERVICE_URI",
    (
        "postgresql+asyncpg://"
        f"{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"
    ),
)
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(CHAT_DATA_DIR / "chat.sqlite3")))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{CHAT_DB_PATH}")
PREDICTION_BACKEND = os.getenv("PREDICTION_BACKEND", "local")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ADK_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", OLLAMA_BASE_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_SKIP_PULL = os.getenv("OLLAMA_SKIP_PULL", "false").lower() == "true"

LLM_BASE_MODEL = os.getenv("LLM_BASE_MODEL", "")
LLM_EPOCHS = int(os.getenv("LLM_EPOCHS", "3"))
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "4"))
LLM_MAX_SEQ_LENGTH = int(os.getenv("LLM_MAX_SEQ_LENGTH", "512"))
LLM_MAX_TRAIN_SAMPLES = _optional_int_env("LLM_MAX_TRAIN_SAMPLES")

FINETUNE_OLLAMA_MODEL = os.getenv("FINETUNE_OLLAMA_MODEL", "tinyllama:1.1b")
FINETUNE_BASE_MODEL = os.getenv(
    "FINETUNE_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
FINETUNE_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "1"))
FINETUNE_BATCH_SIZE = int(os.getenv("FINETUNE_BATCH_SIZE", "1"))
FINETUNE_MAX_SEQ_LENGTH = int(os.getenv("FINETUNE_MAX_SEQ_LENGTH", "256"))
FINETUNE_MAX_TRAIN_SAMPLES = _optional_int_env("FINETUNE_MAX_TRAIN_SAMPLES", "200")
FINETUNE_CLEARML = os.getenv("FINETUNE_CLEARML", "true").lower() == "true"
FINETUNE_AZURE_FILE_ID = os.getenv("FINETUNE_AZURE_FILE_ID", "")
FINETUNE_BEDROCK_DATA_URI = os.getenv("FINETUNE_BEDROCK_DATA_URI", "")
FINETUNE_VERTEX_DATA_URI = os.getenv("FINETUNE_VERTEX_DATA_URI", "")

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-1-8b-instruct-v1:0")
BEDROCK_ROLE_ARN = os.getenv("BEDROCK_ROLE_ARN", "")
BEDROCK_OUTPUT_S3_URI = os.getenv("BEDROCK_OUTPUT_S3_URI", "")
SAGEMAKER_ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME", "ml-ecommerce-chatbot-category"
)

AZURE_LOCATION = os.getenv("AZURE_LOCATION", "westeurope")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-ml-ecommerce-chatbot")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "ecommerce-chat")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_FINE_TUNE_MODEL = os.getenv("AZURE_OPENAI_FINE_TUNE_MODEL", "gpt-4.1-mini")
AZURE_ML_ENDPOINT_NAME = os.getenv(
    "AZURE_ML_ENDPOINT_NAME", "ml-ecommerce-chatbot-category"
)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west4")
VERTEX_MODEL_ID = os.getenv("VERTEX_MODEL_ID", "gemini-2.5-flash")
VERTEX_ENDPOINT_ID = os.getenv(
    "VERTEX_ENDPOINT_ID", "ml-ecommerce-chatbot-prod-category"
)

CLEARML_PROJECT_NAME = os.getenv(
    "CLEARML_PROJECT_NAME", "ML Ecommerce Chatbot/Product Category"
)
CLEARML_ENABLED = os.getenv("CLEARML_ENABLED", "false").lower() == "true"
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST", "http://localhost:8008")
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST", "http://localhost:8080")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST", "http://localhost:8081")
CLEARML_API_ACCESS_KEY = os.getenv(
    "CLEARML_API_ACCESS_KEY",
    os.getenv("CLEARML_ACCESS_KEY", "ml-ecommerce-local"),
)
CLEARML_API_SECRET_KEY = os.getenv(
    "CLEARML_API_SECRET_KEY",
    os.getenv("CLEARML_SECRET_KEY", "ml-ecommerce-local-secret"),
)
CLEARML_SERVING_SERVICE_NAME = os.getenv(
    "CLEARML_SERVING_SERVICE_NAME", "ml-ecommerce-chatbot-serving"
)
CLEARML_SERVING_PROJECT = os.getenv(
    "CLEARML_SERVING_PROJECT", "ML Ecommerce Chatbot/Serving"
)
CLEARML_SERVING_BASE_URL = os.getenv(
    "CLEARML_SERVING_BASE_URL", "http://localhost:8082/serve"
)
CLEARML_SERVING_ENDPOINT = os.getenv("CLEARML_SERVING_ENDPOINT", "product-category")
CLEARML_SERVING_TASK_FILE = Path(
    os.getenv(
        "CLEARML_SERVING_TASK_FILE", str(DATA_DIR / "clearml" / "serving_task_id")
    )
)
CLEARML_SERVING_ENV_FILE = Path(
    os.getenv("CLEARML_SERVING_ENV_FILE", str(DATA_DIR / "clearml" / "serving.env"))
)
CLEARML_LLM_SERVING_ENDPOINT = os.getenv(
    "CLEARML_LLM_SERVING_ENDPOINT", "product-category-llm"
)
CLEARML_LLM_OLLAMA_MODEL = os.getenv("CLEARML_LLM_OLLAMA_MODEL", "")
CLEARML_SERVING_KAFKA_METRIC_SERVER = os.getenv(
    "CLEARML_SERVING_KAFKA_METRIC_SERVER", "clearml-serving-kafka:9092"
)
CLEARML_SERVING_METRIC_LOG_FREQ = float(
    os.getenv("CLEARML_SERVING_METRIC_LOG_FREQ", "1.0")
)

TF_STATE_BUCKET = os.getenv("TF_STATE_BUCKET", "")
TF_STATE_RESOURCE_GROUP = os.getenv("TF_STATE_RESOURCE_GROUP", "")
TF_STATE_STORAGE_ACCOUNT = os.getenv("TF_STATE_STORAGE_ACCOUNT", "")
TF_STATE_CONTAINER = os.getenv("TF_STATE_CONTAINER", "tfstate")
