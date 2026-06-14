"""CatBoost training workflow."""

# Callback and workflow method signatures are dictated by CatBoost and the
# shared training workflow interface.

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from hyperopt import hp

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import DEFAULT_RANDOM_STATE
from src.models.common.base import BaseTrainWorkflow


class ClearMLCatBoostCallback:
    """Report CatBoost native training metrics to ClearML during fitting."""

    def __init__(self, tracker: ClearMLTracker, model_name: str) -> None:
        """Initialize callback.

        Args:
            tracker: Active ClearML tracker.
            model_name: Model key.
        """
        self.tracker = tracker
        self.model_name = model_name

    def after_iteration(self, info: Any) -> bool:
        """Report the latest CatBoost iteration metrics.

        Args:
            info: CatBoost callback info payload.

        Returns:
            Whether training should continue.
        """
        iteration = int(getattr(info, "iteration", 0) or 0)
        metrics = getattr(info, "metrics", {}) or {}
        for dataset_name, dataset_metrics in metrics.items():
            normalized_dataset = CatBoostTrainWorkflow.normalize_dataset_name(
                dataset_name
            )
            for metric_name, values in dataset_metrics.items():
                if not values:
                    continue
                latest_value = values[-1] if isinstance(values, list) else values
                self.tracker.report_scalar_points(
                    f"training_iterations_live/{metric_name}",
                    f"{self.model_name}/{normalized_dataset}",
                    [(iteration, float(latest_value))],
                )
        return True


class CatBoostTrainWorkflow(BaseTrainWorkflow):
    """Train workflow for CatBoost."""

    model_name = "catboost"

    def default_params(self) -> dict[str, Any]:
        """Return default CatBoost parameters.

        Returns:
            Parameter dictionary.
        """
        return {
            "iterations": 350,
            "depth": 5,
            "learning_rate": 0.08,
            "l2_leaf_reg": 3.0,
        }

    def hyperopt_space(self) -> dict[str, Any]:
        """Return Hyperopt search space.

        Returns:
            Hyperopt space.
        """
        return {
            "iterations": hp.quniform("cat_iterations", 150, 600, 50),
            "depth": hp.quniform("cat_depth", 3, 8, 1),
            "learning_rate": hp.loguniform("cat_learning_rate", -3.8, -1.5),
            "l2_leaf_reg": hp.loguniform("cat_l2_leaf_reg", -1.5, 2.0),
        }

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Normalize CatBoost parameters.

        Args:
            params: Raw parameters.

        Returns:
            Normalized parameters.
        """
        normalized = dict(params)
        normalized["iterations"] = int(normalized["iterations"])
        normalized["depth"] = int(normalized["depth"])
        return normalized

    def build_estimator(self, params: dict[str, Any]) -> CatBoostClassifier:
        """Build a CatBoost classifier.

        Args:
            params: Estimator parameters.

        Returns:
            CatBoostClassifier.
        """
        return CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=DEFAULT_RANDOM_STATE,
            verbose=True,
            allow_writing_files=False,
            task_type="GPU",
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
        """Fit CatBoost with validation evaluation history enabled.

        Args:
            pipeline: Unfitted training pipeline.
            x_train: Training features.
            y_train: Training target.
            x_val: Validation features.
            y_val: Validation target.
            tracker: Optional ClearML tracker for live per-iteration logging.

        Returns:
            Native CatBoost evaluation history.
        """
        callbacks = (
            [ClearMLCatBoostCallback(tracker, self.model_name)]
            if tracker and tracker.enabled
            else None
        )
        pipeline.fit(
            x_train,
            y_train,
            classifier__eval_set=(x_val, y_val),
            classifier__verbose=True,
            classifier__callbacks=callbacks,
        )
        classifier = pipeline.named_steps["classifier"]
        return self._normalize_evals_result(classifier.get_evals_result())

    @staticmethod
    def _normalize_evals_result(
        evals_result: dict[str, dict[str, list[float]]],
    ) -> dict[str, dict[str, list[float]]]:
        """Normalize CatBoost eval history names.

        Args:
            evals_result: Raw CatBoost evals result.

        Returns:
            Dataset -> metric -> values payload.
        """
        dataset_names = {
            "learn": "train",
            "validation": "validation",
            "validation_0": "validation",
        }
        history: dict[str, dict[str, list[float]]] = {}
        for dataset_name, metrics in evals_result.items():
            normalized_dataset = dataset_names.get(
                dataset_name,
                CatBoostTrainWorkflow.normalize_dataset_name(dataset_name),
            )
            history[normalized_dataset] = {
                metric_name: [float(value) for value in values]
                for metric_name, values in metrics.items()
            }
        return history

    @staticmethod
    def normalize_dataset_name(dataset_name: str) -> str:
        """Normalize CatBoost dataset aliases.

        Args:
            dataset_name: Raw CatBoost dataset name.

        Returns:
            Normalized dataset name.
        """
        if dataset_name == "learn":
            return "train"
        if dataset_name in {"validation", "validation_0"}:
            return "validation"
        return dataset_name
