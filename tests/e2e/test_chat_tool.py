"""E2E contracts for the ADK prediction tool."""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

from interface.chat.ecommerce_chat.tools import predict_product_category


def test_tool_missing_fields():
    """Test tool gracefully handles missing arguments with ADK error format."""
    result = predict_product_category(
        Age=None,
        Gender=1,
        AnnualIncome=50000,
        NumberOfPurchases=10,
        TimeSpentOnWebsite=120.5,
        LoyaltyProgram=1,
        DiscountsAvailed=2,
        PurchaseStatus=1,
    )
    assert result["status"] == "error"
    assert "CRITICAL: You are missing required fields" in result["message"]


@patch("interface.chat.ecommerce_chat.tools.urllib.request.urlopen")
def test_tool_successful_decoupled_api_call(mock_urlopen):
    """Test the tool properly routes and parses a successful decoupled API call."""
    mock_response = MagicMock()
    # Provide the mocked API response that the FastAPI backend would return
    mock_response.read.return_value = json.dumps(
        {
            "class_id": 4,
            "label": "Sports",
            "confidence": 0.88,
        }
    ).encode("utf-8")

    mock_urlopen.return_value.__enter__.return_value = mock_response

    result = predict_product_category(
        Age=25,
        Gender=0,
        AnnualIncome=60000,
        NumberOfPurchases=15,
        TimeSpentOnWebsite=200.0,
        LoyaltyProgram=0,
        DiscountsAvailed=0,
        PurchaseStatus=1,
    )

    assert "class_id" in result
    assert result["label"] == "Sports"
    assert result["confidence"] == 0.88

    # Verify the request payload serialization
    request_obj = mock_urlopen.call_args[0][0]
    payload = json.loads(request_obj.data.decode("utf-8"))
    assert payload["Age"] == 25
    assert payload["Gender"] == 0
    assert request_obj.method == "POST"


@patch("interface.chat.ecommerce_chat.tools.urllib.request.urlopen")
def test_tool_handles_api_503_error(mock_urlopen):
    """Test the tool gracefully parses 503 errors from the backend."""
    error_io = io.BytesIO(b'{"detail": "No trained model artifact found"}')
    mock_error = urllib.error.HTTPError(
        url="http://localhost:8000/predict",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=error_io,
    )
    mock_urlopen.side_effect = mock_error

    result = predict_product_category(
        Age=25,
        Gender=0,
        AnnualIncome=60000,
        NumberOfPurchases=15,
        TimeSpentOnWebsite=200.0,
        LoyaltyProgram=0,
        DiscountsAvailed=0,
        PurchaseStatus=1,
    )

    assert result["status"] == "error"
    assert result["status_code"] == 503
    assert "No trained model artifact found" in result["message"]
