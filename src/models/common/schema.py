"""Input schema helpers for the ecommerce product-category model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from master_config import FEATURE_COLUMNS


@dataclass(frozen=True)
class CustomerFeatures:
    """Features used by the classifier.

    Args:
        Age: Customer age.
        Gender: Encoded gender value.
        AnnualIncome: Annual income value.
        NumberOfPurchases: Historical number of purchases.
        TimeSpentOnWebsite: Time spent on the website.
        LoyaltyProgram: Loyalty-program membership flag.
        DiscountsAvailed: Number of discounts availed.
        PurchaseStatus: Purchase status flag.
    """

    Age: int
    Gender: int
    AnnualIncome: float
    NumberOfPurchases: int
    TimeSpentOnWebsite: float
    LoyaltyProgram: int
    DiscountsAvailed: int
    PurchaseStatus: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the feature set into a one-row dataframe.

        Returns:
            A dataframe with the exact training feature order.
        """
        return pd.DataFrame([asdict(self)], columns=FEATURE_COLUMNS)


def normalize_gender(value: Any) -> int:
    """Normalize common gender representations into the dataset encoding.

    Args:
        value: Raw gender value from chat, API, or CLI input.

    Returns:
        Integer gender encoding.

    Raises:
        ValueError: If the value cannot be normalized.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _coerce_binary_number(value, "Gender")
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "0": 0,
            "1": 1,
            "female": 0,
            "f": 0,
            "woman": 0,
            "male": 1,
            "m": 1,
            "man": 1,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError("Gender must be 0/1 or a supported gender label")


def normalize_binary(value: Any, field_name: str) -> int:
    """Normalize a flag-like value into 0 or 1.

    Args:
        value: Raw value.
        field_name: Name used in validation errors.

    Returns:
        Integer 0 or 1.

    Raises:
        ValueError: If the value is not a supported binary representation.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _coerce_binary_number(value, field_name)
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "n": 0,
            "y": 1,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"{field_name} must be a binary value")


def _coerce_binary_number(value: int | float, field_name: str) -> int:
    """Normalize a numeric binary value without lossy casts.

    Args:
        value: Numeric value.
        field_name: Name used in validation errors.

    Returns:
        Integer binary value.

    Raises:
        ValueError: If value is not exactly 0 or 1.
    """
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be exactly 0 or 1")
    integer_value = int(value)
    if integer_value not in {0, 1}:
        raise ValueError(f"{field_name} must be exactly 0 or 1")
    return integer_value


def _coerce_int(value: Any, field_name: str) -> int:
    """Normalize an integer field without lossy casts.

    Args:
        value: Raw value.
        field_name: Name used in validation errors.

    Returns:
        Integer value.

    Raises:
        ValueError: If value cannot be represented as an integer.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lstrip("-").isdigit():
            return int(normalized)
    raise ValueError(f"{field_name} must be an integer")


def features_from_mapping(payload: dict[str, Any]) -> CustomerFeatures:
    """Build validated model features from a mapping.

    Args:
        payload: Raw feature payload.

    Returns:
        CustomerFeatures instance.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(f"Missing feature fields: {', '.join(missing)}")

    return CustomerFeatures(
        Age=_coerce_int(payload["Age"], "Age"),
        Gender=normalize_gender(payload["Gender"]),
        AnnualIncome=float(payload["AnnualIncome"]),
        NumberOfPurchases=_coerce_int(
            payload["NumberOfPurchases"], "NumberOfPurchases"
        ),
        TimeSpentOnWebsite=float(payload["TimeSpentOnWebsite"]),
        LoyaltyProgram=normalize_binary(payload["LoyaltyProgram"], "LoyaltyProgram"),
        DiscountsAvailed=_coerce_int(payload["DiscountsAvailed"], "DiscountsAvailed"),
        PurchaseStatus=normalize_binary(payload["PurchaseStatus"], "PurchaseStatus"),
    )


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a feature dataframe from raw records.

    Args:
        records: Raw feature records.

    Returns:
        Dataframe with normalized columns.
    """
    features = [features_from_mapping(record) for record in records]
    return pd.concat(
        [feature.to_dataframe() for feature in features], ignore_index=True
    )
