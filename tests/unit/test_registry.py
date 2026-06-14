"""Tests for model registry."""

from __future__ import annotations

from src.models.registry import parse_model_names


def test_parse_model_names_defaults_to_all() -> None:
    """The all selector expands to all primary model names."""
    assert parse_model_names("all") == [
        "xgboost",
        "randomforest",
        "catboost",
        "logistic_regression",
        "lda",
    ]


def test_parse_model_names_accepts_comma_list() -> None:
    """Comma-separated model names are normalized."""
    assert parse_model_names("xgboost, LDA") == ["xgboost", "lda"]
