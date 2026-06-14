"""XGBoost training workflow."""

# Callback and workflow method signatures are dictated by XGBoost and the
# shared training workflow interface.

from __future__ import annotations

from typing import Any

import pandas as pd
from hyperopt import hp
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import DEFAULT_RANDOM_STATE
from src.models.common.base import BaseTrainWorkflow


class ClearMLXGBoostCallback(TrainingCallback):
    """Report XGBoost native training metrics to ClearML during fitting."""

    def __init__(self, tracker: ClearMLTracker, model_name: str) -> None:
        """Initialize callback.

        Args:
            tracker: Active ClearML tracker.
            model_name: Model key.
        """
        self.tracker = tracker
        self.model_name = model_name

    def after_iteration(
        self,
        model: Any,
        epoch: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> bool:
        """Report the latest iteration metrics.

        Args:
            model: XGBoost booster.
            epoch: Current boosting iteration.
            evals_log: XGBoost evaluation history.

        Returns:
            Whether training should stop.
        """
        for dataset_name, metrics in evals_log.items():
            normalized_dataset = XGBoostTrainWorkflow.normalize_dataset_name(
                dataset_name
            )
            for metric_name, values in metrics.items():
                if not values:
                    continue
                self.tracker.report_scalar_points(
                    f"training_iterations_live/{metric_name}",
                    f"{self.model_name}/{normalized_dataset}",
                    [(epoch, float(values[-1]))],
                )
        return False


class XGBoostTrainWorkflow(BaseTrainWorkflow):
    """Train workflow for XGBoost."""

    model_name = "xgboost"

    def default_params(self) -> dict[str, Any]:
        """Return default XGBoost parameters.

        Returns:
            Parameter dictionary.
        """
        return {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        }

    def hyperopt_space(self) -> dict[str, Any]:
        """Return Hyperopt search space.

        Returns:
            Hyperopt space.
        """
        return {
            "n_estimators": hp.quniform("xgb_n_estimators", 120, 500, 25),
            "max_depth": hp.quniform("xgb_max_depth", 2, 8, 1),
            "learning_rate": hp.loguniform("xgb_learning_rate", -3.5, -1.5),
            "subsample": hp.uniform("xgb_subsample", 0.7, 1.0),
            "colsample_bytree": hp.uniform("xgb_colsample_bytree", 0.7, 1.0),
            "min_child_weight": hp.quniform("xgb_min_child_weight", 1, 8, 1),
            "reg_lambda": hp.loguniform("xgb_reg_lambda", -2.0, 2.0),
        }

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Normalize XGBoost parameters.

        Args:
            params: Raw parameters.

        Returns:
            Normalized parameters.
        """
        normalized = dict(params)
        for key in ["n_estimators", "max_depth", "min_child_weight"]:
            normalized[key] = int(normalized[key])
        return normalized

    def build_estimator(self, params: dict[str, Any]) -> XGBClassifier:
        """Build an XGBoost classifier.

        Args:
            params: Estimator parameters.

        Returns:
            XGBClassifier.
        """
        return XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            random_state=DEFAULT_RANDOM_STATE,
            verbosity=1,
            n_jobs=-1,
        )

    def fit_final_pipeline(
        self,
        pipeline: Any,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        tracker: ClearMLTracker | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        """Fit XGBoost with train/validation evaluation history enabled.

        Args:
            pipeline: Unfitted training pipeline.
            x_train: Training features.
            y_train: Training target.
            x_val: Validation features.
            y_val: Validation target.
            tracker: Optional ClearML tracker for live per-iteration logging.

        Returns:
            Native XGBoost evaluation history.
        """
        classifier = pipeline.named_steps["classifier"]
        callbacks_enabled = tracker is not None and tracker.enabled
        if callbacks_enabled:
            classifier.set_params(
                callbacks=[ClearMLXGBoostCallback(tracker, self.model_name)]
            )
        try:
            pipeline.fit(
                x_train,
                y_train,
                classifier__eval_set=[(x_train, y_train), (x_val, y_val)],
                classifier__verbose=True,
            )
        finally:
            if callbacks_enabled:
                classifier.set_params(callbacks=None)
        return self._normalize_evals_result(classifier.evals_result())

    @staticmethod
    def _normalize_evals_result(
        evals_result: dict[str, dict[str, list[float]]],
    ) -> dict[str, dict[str, list[float]]]:
        """Normalize XGBoost eval history names.

        Args:
            evals_result: Raw XGBoost evals result.

        Returns:
            Dataset -> metric -> values payload.
        """
        dataset_names = {
            "validation_0": "train",
            "validation_1": "validation",
        }
        history: dict[str, dict[str, list[float]]] = {}
        for dataset_name, metrics in evals_result.items():
            normalized_dataset = dataset_names.get(
                dataset_name,
                XGBoostTrainWorkflow.normalize_dataset_name(dataset_name),
            )
            history[normalized_dataset] = {
                metric_name: [float(value) for value in values]
                for metric_name, values in metrics.items()
            }
        return history

    @staticmethod
    def normalize_dataset_name(dataset_name: str) -> str:
        """Normalize XGBoost dataset aliases.

        Args:
            dataset_name: Raw XGBoost dataset name.

        Returns:
            Normalized dataset name.
        """
        if dataset_name == "validation_0":
            return "train"
        if dataset_name == "validation_1":
            return "validation"
        return dataset_name
