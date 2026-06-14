"""Model workflow registry."""

from __future__ import annotations

from src.models.catboost_model import CatBoostTrainWorkflow
from src.models.common.base import BaseTrainWorkflow
from src.models.lda import LDATrainWorkflow
from src.models.logistic_regression import LogisticRegressionTrainWorkflow
from src.models.random_forest import RandomForestTrainWorkflow
from src.models.xgboost_model import XGBoostTrainWorkflow

MODEL_WORKFLOWS: dict[str, type[BaseTrainWorkflow]] = {
    "xgboost": XGBoostTrainWorkflow,
    "randomforest": RandomForestTrainWorkflow,
    "catboost": CatBoostTrainWorkflow,
    "logistic_regression": LogisticRegressionTrainWorkflow,
    "logreg": LogisticRegressionTrainWorkflow,
    "lda": LDATrainWorkflow,
}

PRIMARY_MODEL_NAMES = [
    "xgboost",
    "randomforest",
    "catboost",
    "logistic_regression",
    "lda",
]


def get_workflow(model_name: str) -> BaseTrainWorkflow:
    """Create a train workflow by model name.

    Args:
        model_name: Model key.

    Returns:
        Train workflow instance.

    Raises:
        ValueError: If the model is unknown.
    """
    key = model_name.lower()
    if key not in MODEL_WORKFLOWS:
        available = ", ".join(sorted(MODEL_WORKFLOWS))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_WORKFLOWS[key]()


def parse_model_names(raw_models: str | None) -> list[str]:
    """Parse model selector text.

    Args:
        raw_models: Comma-separated model list or "all".

    Returns:
        Normalized model keys.
    """
    if raw_models is None or raw_models.strip().lower() == "all":
        return PRIMARY_MODEL_NAMES
    return [model.strip().lower() for model in raw_models.split(",") if model.strip()]
