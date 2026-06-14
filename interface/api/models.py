"""Pydantic request models for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Prediction request body."""

    Age: int = Field(..., ge=0, le=120)
    Gender: int | str
    AnnualIncome: float = Field(..., ge=0)
    NumberOfPurchases: int = Field(..., ge=0)
    TimeSpentOnWebsite: float = Field(..., ge=0)
    LoyaltyProgram: int | bool | str
    DiscountsAvailed: int = Field(..., ge=0)
    PurchaseStatus: int | bool | str
