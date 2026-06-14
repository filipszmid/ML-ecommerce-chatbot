"""Unit tests for model artifact loading and prediction mapping."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.common.schema import CustomerFeatures
from src.models.common.metrics import probabilities_to_labels
from src.models.inference import ProductCategoryPredictor


@pytest.fixture(name="mock_sklearn_pipeline")
def make_mock_sklearn_pipeline():
    """Mock a scikit-learn pipeline."""
    pipeline = MagicMock()
    pipeline.classes_ = np.array([0, 1, 2, 3, 4])
    pipeline.predict_proba.return_value = np.array([[0.1, 0.6, 0.1, 0.1, 0.1]])
    return pipeline


@patch("src.models.inference.latest_model_dir")
def test_predictor_raises_not_found(mock_latest):
    """Test predictor raises FileNotFoundError when no models exist."""
    mock_latest.return_value = None
    with pytest.raises(FileNotFoundError, match="No trained model artifact found"):
        ProductCategoryPredictor()


@patch("src.models.inference.joblib.load")
@patch("src.models.inference.latest_model_dir")
def test_predictor_successful_inference(
    mock_latest, mock_load, mock_sklearn_pipeline, tmp_path
):
    """Test successful model prediction mapping."""
    # Setup mock artifact directory
    artifact_dir = tmp_path / "runs" / "test_run"
    artifact_dir.mkdir(parents=True)

    # Write mock metadata and dummy model file
    metadata = {"run_id": "test_run", "model_name": "test_model"}
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata))
    (artifact_dir / "model.joblib").write_text("dummy")

    # Configure mocks
    mock_latest.return_value = artifact_dir
    mock_load.return_value = mock_sklearn_pipeline

    # Initialize predictor
    predictor = ProductCategoryPredictor(artifact_dir=str(artifact_dir))

    features = CustomerFeatures(
        Age=30,
        Gender=1,
        AnnualIncome=50000.0,
        NumberOfPurchases=10,
        TimeSpentOnWebsite=120.5,
        LoyaltyProgram=1,
        DiscountsAvailed=2,
        PurchaseStatus=1,
    )

    result = predictor.predict_one(features)

    assert result["class_id"] == 1
    assert result["label"] == "Clothing"
    assert result["confidence"] == 0.6
    assert result["model_name"] == "test_model"
    assert result["run_id"] == "test_run"

    # Verify probability distribution
    probs = result["probabilities"]
    assert len(probs) == 5
    assert probs[0]["class_id"] == 1
    assert probs[0]["probability"] == 0.6


def test_probabilities_to_labels_uses_estimator_classes() -> None:
    """Probability columns should be mapped with fitted class labels."""
    probs = probabilities_to_labels(
        np.array([[0.7, 0.3]]),
        class_labels=np.array([2, 4]),
    )

    assert probs[0]["class_id"] == 2
    assert probs[0]["label"] == "Home Goods"
    assert probs[1]["class_id"] == 4
    assert probs[1]["label"] == "Sports"
