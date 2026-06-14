"""ADK tool functions for the ecommerce chat agent."""

# ADK exposes Python function argument names to the model. These names mirror
# the trained tabular feature schema and must stay PascalCase.

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from master_config import API_URL


def predict_product_category(
    Age: int,
    Gender: int,
    AnnualIncome: float,
    NumberOfPurchases: int,
    TimeSpentOnWebsite: float,
    LoyaltyProgram: int,
    DiscountsAvailed: int,
    PurchaseStatus: int,
) -> dict[str, Any]:
    """Predict product category by calling the prediction API endpoint.

    Args:
        Age: Customer age.
        Gender: Encoded gender value.
        AnnualIncome: Annual income.
        NumberOfPurchases: Historical purchase count.
        TimeSpentOnWebsite: Time spent on website.
        LoyaltyProgram: Loyalty-program flag.
        DiscountsAvailed: Number of discounts used.
        PurchaseStatus: Purchase completion flag.

    Returns:
        API prediction payload or an error payload.
    """
    if any(
        v is None
        for v in (
            Age,
            Gender,
            AnnualIncome,
            NumberOfPurchases,
            TimeSpentOnWebsite,
            LoyaltyProgram,
            DiscountsAvailed,
            PurchaseStatus,
        )
    ):
        return {
            "status": "error",
            "message": (
                "CRITICAL: You are missing required fields. You MUST ask the user "
                "for the missing information before calling this tool. Do NOT "
                "guess or pass null/None."
            ),
        }

    payload = {
        "Age": Age,
        "Gender": Gender,
        "AnnualIncome": AnnualIncome,
        "NumberOfPurchases": NumberOfPurchases,
        "TimeSpentOnWebsite": TimeSpentOnWebsite,
        "LoyaltyProgram": LoyaltyProgram,
        "DiscountsAvailed": DiscountsAvailed,
        "PurchaseStatus": PurchaseStatus,
    }
    api_url = API_URL.rstrip("/")
    request = urllib.request.Request(
        url=f"{api_url}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "status": "error",
            "message": exc.read().decode("utf-8"),
            "status_code": exc.code,
        }
    except (json.JSONDecodeError, OSError, TimeoutError, urllib.error.URLError) as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
