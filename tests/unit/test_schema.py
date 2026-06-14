"""Tests for feature schema normalization."""

from __future__ import annotations

import pytest

from src.models.common.schema import features_from_mapping


def test_features_from_mapping_normalizes_human_values() -> None:
    """Feature mapping accepts chat-style values."""
    features = features_from_mapping(
        {
            "Age": "40",
            "Gender": "male",
            "AnnualIncome": "65000.5",
            "NumberOfPurchases": "8",
            "TimeSpentOnWebsite": "31.2",
            "LoyaltyProgram": "yes",
            "DiscountsAvailed": "2",
            "PurchaseStatus": "true",
        }
    )

    assert features.Gender == 1
    assert features.LoyaltyProgram == 1
    assert features.PurchaseStatus == 1
    assert list(features.to_dataframe().columns) == [
        "Age",
        "Gender",
        "AnnualIncome",
        "NumberOfPurchases",
        "TimeSpentOnWebsite",
        "LoyaltyProgram",
        "DiscountsAvailed",
        "PurchaseStatus",
    ]


def test_features_from_mapping_rejects_invalid_gender() -> None:
    """Gender must stay in the trained binary encoding."""
    with pytest.raises(ValueError, match="Gender"):
        features_from_mapping(
            {
                "Age": "40",
                "Gender": 9,
                "AnnualIncome": "65000.5",
                "NumberOfPurchases": "8",
                "TimeSpentOnWebsite": "31.2",
                "LoyaltyProgram": "yes",
                "DiscountsAvailed": "2",
                "PurchaseStatus": "true",
            }
        )


def test_features_from_mapping_rejects_lossy_binary_cast() -> None:
    """Binary fields should not silently cast fractional values."""
    with pytest.raises(ValueError, match="LoyaltyProgram"):
        features_from_mapping(
            {
                "Age": "40",
                "Gender": "male",
                "AnnualIncome": "65000.5",
                "NumberOfPurchases": "8",
                "TimeSpentOnWebsite": "31.2",
                "LoyaltyProgram": 0.8,
                "DiscountsAvailed": "2",
                "PurchaseStatus": "true",
            }
        )
