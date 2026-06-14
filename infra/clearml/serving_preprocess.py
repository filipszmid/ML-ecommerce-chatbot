"""Pre/post-processing for ClearML Serving product-category endpoint."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np

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


class Preprocess:
    """ClearML Serving adapter for the serialized sklearn pipeline."""

    def load(self, local_model_file: str) -> Any:
        """Load the serialized model.

        Args:
            local_model_file: Local model artifact path.

        Returns:
            Loaded model pipeline.
        """

        self._model = joblib.load(local_model_file)
        return self._model

    def preprocess(
        self, request: dict[str, Any], state: dict[str, Any], _
    ) -> np.ndarray:
        """Convert JSON request payload to a model input matrix.

        Args:
            request: Incoming request body.
            state: Request state.
            _: Optional custom statistics callback.

        Returns:
            Feature matrix.
        """

        payload = request.get("features", request)
        row = [float(payload[column]) for column in FEATURE_COLUMNS]
        state["features"] = dict(zip(FEATURE_COLUMNS, row, strict=True))
        return np.asarray([row], dtype=np.float32)

    def process(self, data: np.ndarray, state: dict[str, Any], _) -> dict[str, Any]:
        """Run prediction with the loaded model.

        Args:
            data: Feature matrix.
            state: Request state.
            _: Optional custom statistics callback.

        Returns:
            Raw prediction payload.
        """

        probabilities = self._model.predict_proba(data)[0]
        class_labels = self._class_labels()
        predicted_id = int(class_labels[int(np.argmax(probabilities))])
        return {
            "predicted_class_id": predicted_id,
            "predicted_category": PRODUCT_CATEGORY_LABELS[predicted_id],
            "probabilities": [
                {
                    "class_id": int(class_id),
                    "label": PRODUCT_CATEGORY_LABELS[int(class_id)],
                    "probability": float(probability),
                }
                for class_id, probability in zip(
                    class_labels, probabilities, strict=True
                )
            ],
            "features": state.get("features", {}),
        }

    def postprocess(self, data: Any, state: dict[str, Any], _) -> dict[str, Any]:
        """Return the API response body.

        Args:
            data: Prediction payload or sklearn prediction array.
            state: Request state.
            _: Optional custom statistics callback.

        Returns:
            Response payload.
        """

        if isinstance(data, dict):
            return data

        prediction = data.tolist() if isinstance(data, np.ndarray) else data
        predicted_id = int(
            prediction[0] if isinstance(prediction, list) else prediction
        )
        return {
            "predicted_class_id": predicted_id,
            "predicted_category": PRODUCT_CATEGORY_LABELS[predicted_id],
            "prediction": prediction,
            "features": state.get("features", {}),
        }

    def _class_labels(self) -> list[int]:
        """Return fitted class labels matching probability columns.

        Returns:
            Class labels from the model or classifier pipeline step.
        """

        if hasattr(self._model, "classes_"):
            return [int(class_id) for class_id in self._model.classes_]
        named_steps = getattr(self._model, "named_steps", {})
        classifier = named_steps.get("classifier") if named_steps else None
        if classifier is not None and hasattr(classifier, "classes_"):
            return [int(class_id) for class_id in classifier.classes_]
        return list(range(len(PRODUCT_CATEGORY_LABELS)))
