"""Prediction Click commands."""

from __future__ import annotations

import json
from typing import Any

import click

from interface.cli.utils import HELP_CONTEXT, echo_json
from src.models.common.schema import features_from_mapping
from src.models.inference import ProductCategoryPredictor


def run_prediction(model_dir: str | None, json_payload: str) -> dict[str, Any]:
    """Run one product-category prediction.

    Args:
        model_dir: Optional model artifact directory.
        json_payload: JSON feature payload.

    Returns:
        Prediction payload.
    """
    try:
        payload = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(str(exc), param_hint="--json") from exc

    features = features_from_mapping(payload)
    predictor = ProductCategoryPredictor(artifact_dir=model_dir)
    return predictor.predict_one(features)


@click.command(
    name="predict", context_settings=HELP_CONTEXT, help="Predict from JSON input."
)
@click.option("--model-dir", default=None)
@click.option("--json", "json_payload", required=True)
def predict_command(model_dir: str | None, json_payload: str) -> None:
    """Run the prediction command.

    Args:
        model_dir: Optional model artifact directory.
        json_payload: JSON feature payload.
    """
    echo_json(run_prediction(model_dir=model_dir, json_payload=json_payload))
