"""FastAPI app for product-category prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from interface.api.models import PredictionRequest
from master_config import (
    ADK_WEB_URL,
    CORS_ALLOW_ORIGINS,
    MODEL_ARTIFACT_DIR,
    PRODUCT_CATEGORY_LABELS,
)
from src.models.common.artifacts import latest_model_dir
from src.models.common.schema import features_from_mapping
from src.models.inference import ProductCategoryPredictor

app = FastAPI(
    title="ML Ecommerce Chatbot",
    version="0.1.0",
    description="Product-category prediction backend for the ADK chat agent.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor_cache: dict[str, Any] = {
    "artifact_dir": None,
    "model_mtime_ns": None,
    "predictor": None,
    "predictor_class": None,
}


@app.get("/health")
def health() -> dict[str, str]:
    """Return health status.

    Returns:
        Health payload.
    """
    return {"status": "ok"}


@app.get("/")
def chat_index() -> RedirectResponse:
    """Redirect to the ADK Web chat UI.

    Returns:
        Redirect response.
    """
    return RedirectResponse(url=ADK_WEB_URL)


@app.get("/models/latest")
def get_latest_model() -> dict[str, Any]:
    """Return latest local model artifact info.

    Returns:
        Model metadata.
    """
    model_dir = latest_model_dir(MODEL_ARTIFACT_DIR)
    if model_dir is None:
        return {"available": False, "artifact_dir": None}
    return {"available": True, "artifact_dir": str(model_dir)}


@app.get("/categories")
def categories() -> dict[int, str]:
    """Return product-category labels.

    Returns:
        Category mapping.
    """
    return PRODUCT_CATEGORY_LABELS


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    """Predict a product category.

    Args:
        payload: Prediction request.

    Returns:
        Prediction payload.
    """
    try:
        features = features_from_mapping(payload.model_dump())
        return _get_predictor().predict_one(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _get_predictor() -> ProductCategoryPredictor:
    """Return a cached predictor for the current latest model artifact.

    Returns:
        Cached or newly loaded predictor.

    Raises:
        FileNotFoundError: If no trained model is available.
    """
    model_dir = latest_model_dir(MODEL_ARTIFACT_DIR)
    model_mtime_ns = _model_mtime_ns(model_dir)

    if (
        _predictor_cache["predictor"] is None
        or _predictor_cache["artifact_dir"] != model_dir
        or _predictor_cache["model_mtime_ns"] != model_mtime_ns
        or _predictor_cache["predictor_class"] is not ProductCategoryPredictor
    ):
        _predictor_cache["predictor"] = ProductCategoryPredictor(artifact_dir=model_dir)
        _predictor_cache["artifact_dir"] = model_dir
        _predictor_cache["model_mtime_ns"] = model_mtime_ns
        _predictor_cache["predictor_class"] = ProductCategoryPredictor
    return _predictor_cache["predictor"]


def reset_predictor_cache() -> None:
    """Reset the predictor cache.

    This is used by tests and local reload workflows that need deterministic
    cache state after monkeypatching predictor dependencies.
    """
    _predictor_cache.update(
        {
            "artifact_dir": None,
            "model_mtime_ns": None,
            "predictor": None,
            "predictor_class": None,
        }
    )


def _model_mtime_ns(model_dir: Path | None) -> int | None:
    """Return model artifact mtime for cache invalidation.

    Args:
        model_dir: Model artifact directory.

    Returns:
        Model file mtime in nanoseconds, if available.
    """
    if model_dir is None:
        return None
    try:
        return (model_dir / "model.joblib").stat().st_mtime_ns
    except FileNotFoundError:
        return None
