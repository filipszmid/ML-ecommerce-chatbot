"""Random Forest training workflow."""

from __future__ import annotations

from typing import Any

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier

from master_config import DEFAULT_RANDOM_STATE
from src.models.common.base import BaseTrainWorkflow


class RandomForestTrainWorkflow(BaseTrainWorkflow):
    """Train workflow for Random Forest."""

    model_name = "randomforest"

    def default_params(self) -> dict[str, Any]:
        """Return default Random Forest parameters.

        Returns:
            Parameter dictionary.
        """
        return {
            "n_estimators": 350,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        }

    def hyperopt_space(self) -> dict[str, Any]:
        """Return Hyperopt search space.

        Returns:
            Hyperopt space.
        """
        return {
            "n_estimators": hp.quniform("rf_n_estimators", 150, 650, 50),
            "max_depth": hp.choice("rf_max_depth", [None, 4, 6, 8, 12, 16]),
            "min_samples_split": hp.quniform("rf_min_samples_split", 2, 12, 1),
            "min_samples_leaf": hp.quniform("rf_min_samples_leaf", 1, 8, 1),
            "class_weight": hp.choice(
                "rf_class_weight", ["balanced", "balanced_subsample"]
            ),
        }

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Normalize Random Forest parameters.

        Args:
            params: Raw parameters.

        Returns:
            Normalized parameters.
        """
        normalized = dict(params)
        for key in ["n_estimators", "min_samples_split", "min_samples_leaf"]:
            normalized[key] = int(normalized[key])
        if normalized["max_depth"] is not None:
            normalized["max_depth"] = int(normalized["max_depth"])
        return normalized

    def build_estimator(self, params: dict[str, Any]) -> RandomForestClassifier:
        """Build a Random Forest classifier.

        Args:
            params: Estimator parameters.

        Returns:
            RandomForestClassifier.
        """
        return RandomForestClassifier(
            **params,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
