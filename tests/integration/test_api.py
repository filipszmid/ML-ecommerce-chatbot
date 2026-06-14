"""Integration tests for the FastAPI prediction service."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from interface.api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_categories_endpoint():
    """Test the categories endpoint."""
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert "0" in data
    assert data["0"] == "Electronics"


@patch("interface.api.app.ProductCategoryPredictor")
def test_predict_endpoint_success(mock_predictor_class):
    """Test successful prediction endpoint."""
    # Setup mock predictor
    mock_instance = mock_predictor_class.return_value
    mock_instance.predict_one.return_value = {
        "class_id": 1,
        "label": "Clothing",
        "confidence": 0.95,
    }

    payload = {
        "Age": 30,
        "Gender": 1,
        "AnnualIncome": 50000,
        "NumberOfPurchases": 10,
        "TimeSpentOnWebsite": 120.5,
        "LoyaltyProgram": 1,
        "DiscountsAvailed": 2,
        "PurchaseStatus": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["class_id"] == 1
    assert data["label"] == "Clothing"
    assert data["confidence"] == 0.95

    mock_instance.predict_one.assert_called_once()


def test_predict_endpoint_validation_error():
    """Test payload validation (e.g., negative Age)."""
    payload = {
        "Age": -5,  # Invalid
        "Gender": 1,
        "AnnualIncome": 50000,
        "NumberOfPurchases": 10,
        "TimeSpentOnWebsite": 120.5,
        "LoyaltyProgram": 1,
        "DiscountsAvailed": 2,
        "PurchaseStatus": 1,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@patch("interface.api.app.ProductCategoryPredictor")
def test_predict_endpoint_no_model(mock_predictor_class):
    """Test prediction when no model is found."""
    mock_predictor_class.side_effect = FileNotFoundError(
        "No trained model artifact found"
    )

    payload = {
        "Age": 30,
        "Gender": 1,
        "AnnualIncome": 50000,
        "NumberOfPurchases": 10,
        "TimeSpentOnWebsite": 120.5,
        "LoyaltyProgram": 1,
        "DiscountsAvailed": 2,
        "PurchaseStatus": 1,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "No trained model artifact found" in response.json()["detail"]
