"""Tests for SHAP helper model classification."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.common.metrics import _is_tree_model


def test_tree_model_detection_for_supported_estimators() -> None:
    """Tree models should use the SHAP TreeExplainer path."""
    assert _is_tree_model(RandomForestClassifier()) is True


def test_tree_model_detection_for_linear_estimators() -> None:
    """Non-tree models should use the generic SHAP explainer path."""
    assert _is_tree_model(LogisticRegression()) is False
