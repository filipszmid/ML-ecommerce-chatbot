"""Inference helpers for trained model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from master_config import (
    DATA_DIR,
    MODEL_ARTIFACT_DIR,
    PRODUCT_CATEGORY_LABELS,
    PROJECT_ROOT,
)
from src.models.common.artifacts import latest_model_dir, load_json
from src.models.common.metrics import probabilities_to_labels
from src.models.common.schema import CustomerFeatures, dataframe_from_records


class ProductCategoryPredictor:
    """Load and serve a trained product-category classifier."""

    def __init__(self, artifact_dir: Path | str | None = None) -> None:
        """Initialize predictor.

        Args:
            artifact_dir: Model artifact directory. If omitted, latest run is used.

        Raises:
            FileNotFoundError: If no trained model can be found.
        """
        selected_dir = (
            Path(artifact_dir) if artifact_dir else latest_model_dir(MODEL_ARTIFACT_DIR)
        )
        if selected_dir is None:
            raise FileNotFoundError(
                "No trained model artifact found in data/runs/classical_ml"
            )
        if not selected_dir.is_absolute():
            parts = selected_dir.parts
            if parts and parts[0] == "data":
                selected_dir = (PROJECT_ROOT / selected_dir).resolve()
            else:
                selected_dir = (DATA_DIR / selected_dir).resolve()
        self.artifact_dir = selected_dir
        model_path = self.artifact_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        self.model = joblib.load(model_path)
        metadata_path = self.artifact_dir / "metadata.json"
        self.metadata = load_json(metadata_path) if metadata_path.exists() else {}

    def predict_one(self, features: CustomerFeatures) -> dict[str, Any]:
        """Predict a single feature payload.

        Args:
            features: Customer features.

        Returns:
            Prediction payload.
        """
        dataframe = features.to_dataframe()
        predicted_class = int(self.model.predict(dataframe)[0])
        probabilities = []
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = probabilities_to_labels(
                self.model.predict_proba(dataframe),
                class_labels=self._class_labels(),
            )
            confidence = (
                float(probabilities[0]["probability"]) if probabilities else None
            )
        return {
            "class_id": predicted_class,
            "label": PRODUCT_CATEGORY_LABELS.get(predicted_class, str(predicted_class)),
            "confidence": confidence,
            "probabilities": probabilities,
            "artifact_dir": str(self.artifact_dir),
            "model_name": self.metadata.get("model_name"),
            "run_id": self.metadata.get("run_id"),
        }

    def predict_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict a batch of raw records.

        Args:
            records: Raw feature records.

        Returns:
            Prediction payloads.
        """
        dataframe = dataframe_from_records(records)
        predictions = self.model.predict(dataframe)
        probabilities = (
            self.model.predict_proba(dataframe)
            if hasattr(self.model, "predict_proba")
            else None
        )
        outputs = []
        for index, prediction in enumerate(predictions):
            predicted_class = int(prediction)
            probability_payload = []
            confidence = None
            if probabilities is not None:
                probability_payload = probabilities_to_labels(
                    probabilities[index: index + 1],
                    class_labels=self._class_labels(),
                )
                confidence = (
                    float(probability_payload[0]["probability"])
                    if probability_payload
                    else None
                )
            outputs.append(
                {
                    "class_id": predicted_class,
                    "label": PRODUCT_CATEGORY_LABELS.get(
                        predicted_class, str(predicted_class)
                    ),
                    "confidence": confidence,
                    "probabilities": probability_payload,
                }
            )
        return outputs

    def _class_labels(self) -> list[int] | None:
        """Return fitted class labels matching `predict_proba` columns.

        Returns:
            Class labels, if exposed by the model or classifier step.
        """
        if hasattr(self.model, "classes_"):
            return [int(class_id) for class_id in self.model.classes_]
        named_steps = getattr(self.model, "named_steps", {})
        classifier = named_steps.get("classifier") if named_steps else None
        if classifier is not None and hasattr(classifier, "classes_"):
            return [int(class_id) for class_id in classifier.classes_]
        return None
