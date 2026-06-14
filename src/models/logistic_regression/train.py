"""Logistic Regression training workflow."""

from __future__ import annotations

from typing import Any

from hyperopt import hp
from sklearn.linear_model import LogisticRegression

from master_config import DEFAULT_RANDOM_STATE
from src.models.common.base import BaseTrainWorkflow


class LogisticRegressionTrainWorkflow(BaseTrainWorkflow):
    """Train workflow for multinomial Logistic Regression."""

    model_name = "logistic_regression"
    requires_scaling = True

    def default_params(self) -> dict[str, Any]:
        """Return default Logistic Regression parameters.

        Returns:
            Parameter dictionary.
        """
        return {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        }

    def hyperopt_space(self) -> dict[str, Any]:
        """Return Hyperopt search space.

        Returns:
            Hyperopt space.
        """
        return {
            "C": hp.loguniform("lr_c", -4.0, 3.0),
            "class_weight": hp.choice("lr_class_weight", ["balanced", None]),
        }

    def build_estimator(self, params: dict[str, Any]) -> LogisticRegression:
        """Build a Logistic Regression classifier.

        Args:
            params: Estimator parameters.

        Returns:
            LogisticRegression.
        """
        merged = {**self.default_params(), **params}
        return LogisticRegression(
            **merged,
            random_state=DEFAULT_RANDOM_STATE,
            verbose=1,
        )
