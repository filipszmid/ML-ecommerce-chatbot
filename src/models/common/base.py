"""Base workflow for classical model training."""

# This module owns the stable training workflow interface and orchestration
# path. The public method signatures are used by CLI, ClearML, and tests.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import (
    CLEARML_ENABLED,
    DEFAULT_CV_FOLDS,
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_EVALS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SMOTE_K_NEIGHBORS,
    DEFAULT_TEST_SIZE,
    DEFAULT_USE_SMOTE,
    EVALS_DIR,
    FEATURE_COLUMNS,
    MODEL_SELECTION_METRIC,
    PRODUCT_CATEGORY_LABELS,
    TARGET_COLUMN,
)
from src.models.common.artifacts import (
    ensure_artifact_dirs,
    feature_schema,
    make_run_id,
    run_dir,
    save_json,
    write_training_report,
)
from src.models.common.data import (
    load_dataset,
    split_features_target,
    train_validation_split,
)
from src.models.common.metrics import (
    confusion_matrix_to_markdown,
    evaluate_classifier,
    generate_shap_summary,
)


@dataclass(frozen=True)
class TrainingResult:
    """Result of a training workflow.

    Args:
        model_name: Model key.
        run_id: Run identifier.
        artifact_dir: Directory with artifacts.
        metrics: Validation metrics.
        params: Best/final parameters.
        report_path: Markdown report path.
    """

    model_name: str
    run_id: str
    artifact_dir: Path
    metrics: dict[str, float]
    params: dict[str, Any]
    report_path: Path
    clearml_model_id: str | None = None
    clearml_task_id: str | None = None


class BaseTrainWorkflow(ABC):
    """Base workflow shared by model-specific trainers."""

    model_name: str
    requires_scaling: bool = False

    def __init__(
        self,
        random_state: int = DEFAULT_RANDOM_STATE,
        selection_metric: str = MODEL_SELECTION_METRIC,
    ) -> None:
        """Initialize workflow.

        Args:
            random_state: Random seed.
            selection_metric: Metric used for model selection.
        """
        self.random_state = random_state
        self.selection_metric = selection_metric

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default estimator parameters.

        Returns:
            Default parameters.
        """

    @abstractmethod
    def hyperopt_space(self) -> dict[str, Any]:
        """Return Hyperopt search space.

        Returns:
            Hyperopt parameter space.
        """

    @abstractmethod
    def build_estimator(self, params: dict[str, Any]) -> Any:
        """Build an unfitted estimator.

        Args:
            params: Estimator parameters.

        Returns:
            Unfitted estimator.
        """

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Normalize Hyperopt parameter values.

        Args:
            params: Raw parameter dictionary.

        Returns:
            Normalized parameter dictionary.
        """
        return params

    def build_pipeline(
        self,
        params: dict[str, Any],
        use_smote: bool = False,
        smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
    ) -> Pipeline | ImbalancedPipeline:
        """Build a scikit-learn pipeline for the estimator.

        Args:
            params: Estimator parameters.
            use_smote: Whether to add SMOTE to the training pipeline.
            smote_k_neighbors: SMOTE neighbor count.

        Returns:
            Pipeline.
        """
        estimator = self.build_estimator(params)
        steps: list[tuple[str, Any]] = []
        if self.requires_scaling:
            steps.append(("scaler", StandardScaler()))
        if use_smote:
            steps.append(
                (
                    "smote",
                    SMOTE(
                        random_state=self.random_state,
                        k_neighbors=smote_k_neighbors,
                    ),
                )
            )
        steps.append(("classifier", estimator))
        if use_smote:
            return ImbalancedPipeline(steps)
        return Pipeline(steps)

    def run(
        self,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        run_name: str | None = None,
        max_evals: int = DEFAULT_MAX_EVALS,
        cv_folds: int = DEFAULT_CV_FOLDS,
        use_smote: bool = DEFAULT_USE_SMOTE,
        smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
        clearml_enabled: bool = CLEARML_ENABLED,
        pipeline_id: str | None = None,
        clearml_dataset_id: str | None = None,
        clearml_tracker: ClearMLTracker | None = None,
    ) -> TrainingResult:
        """Run hyperparameter selection, final training, and reporting.

        Args:
            dataset_path: Training CSV path.
            run_name: Optional fixed run identifier.
            max_evals: Hyperopt evaluations. Use 0 for defaults only.
            cv_folds: Stratified folds used during hyperparameter selection.
            use_smote: Whether to use SMOTE during training and CV.
            smote_k_neighbors: SMOTE neighbor count.
            clearml_enabled: Whether to log into ClearML.
            pipeline_id: Optional parent model-selection identifier.
            clearml_dataset_id: Optional ClearML dataset id from a parent workflow.
            clearml_tracker: Optional existing ClearML task for parent workflows.

        Returns:
            Training result.
        """
        ensure_artifact_dirs()
        dataset_path = Path(dataset_path)
        run_id = run_name or make_run_id(self.model_name)
        artifact_dir = run_dir(run_id, pipeline_id=pipeline_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.add(artifact_dir / "training.log")
        logger.info(f"Starting training workflow for {self.model_name}")

        dataframe = load_dataset(dataset_path)
        x_train, x_val, y_train, y_val = train_validation_split(
            dataframe=dataframe,
            random_state=self.random_state,
        )

        train_indices = x_train.index.tolist()
        test_indices = x_val.index.tolist()

        best_params = self.default_params()
        trials_payload: list[dict[str, Any]] = []
        if max_evals > 0:
            best_params, trials_payload = self._run_hyperopt(
                dataframe=dataframe,
                max_evals=max_evals,
                cv_folds=cv_folds,
                use_smote=use_smote,
                smote_k_neighbors=smote_k_neighbors,
            )

        owns_tracker = clearml_tracker is None
        tracker = clearml_tracker or ClearMLTracker(enabled=clearml_enabled)
        if owns_tracker:
            tracker.start(
                task_name=run_id,
                params={
                    "model_name": self.model_name,
                    "dataset_path": str(dataset_path),
                    "pipeline_id": pipeline_id,
                    "clearml_dataset_id": clearml_dataset_id,
                    "max_evals": max_evals,
                    "cv_folds": cv_folds,
                    "use_smote": use_smote,
                    "smote_k_neighbors": smote_k_neighbors,
                    "params": best_params,
                },
                task_type="training",
                tags=[
                    "classical-ml",
                    "model-training",
                    f"model:{self.model_name}",
                    *(["model-selection-child"] if pipeline_id else []),
                ],
            )
        if clearml_dataset_id is None and pipeline_id is None:
            clearml_dataset_id = tracker.log_dataset(
                dataset_path=dataset_path,
                dataset_name="customer_purchase_data",
                dataset_version=run_id,
                tags=["tabular", "ecommerce", "training"],
            )

        pipeline = self.build_pipeline(
            best_params,
            use_smote=use_smote,
            smote_k_neighbors=smote_k_neighbors,
        )
        training_history = self.fit_final_pipeline(
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            tracker=tracker,
        )
        metrics, report, matrix = evaluate_classifier(pipeline, x_val, y_val)

        joblib.dump(pipeline, artifact_dir / "model.joblib")
        save_json(artifact_dir / "metrics.json", metrics)
        save_json(artifact_dir / "params.json", best_params)
        save_json(artifact_dir / "feature_schema.json", feature_schema())
        save_json(artifact_dir / "train_indices.json", train_indices)
        save_json(artifact_dir / "test_indices.json", test_indices)
        save_json(artifact_dir / "val_indices.json", test_indices)
        split_manifest = self._split_manifest(
            dataset_path=dataset_path,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        save_json(artifact_dir / "split_manifest.json", split_manifest)
        save_json(
            artifact_dir / "metadata.json",
            {
                "model_name": self.model_name,
                "run_id": run_id,
                "selection_metric": self.selection_metric,
                "dataset_path": str(dataset_path),
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "use_smote": use_smote,
                "smote_k_neighbors": smote_k_neighbors,
                "split_manifest_path": "split_manifest.json",
            },
        )
        save_json(artifact_dir / "hyperopt_trials.json", trials_payload)
        save_json(artifact_dir / "training_history.json", training_history)
        matrix.to_csv(artifact_dir / "confusion_matrix.csv")
        (artifact_dir / "classification_report.txt").write_text(report)

        # SHAP Analysis
        shap_image_path = generate_shap_summary(
            pipeline=pipeline,
            x_val=x_val,
            feature_columns=FEATURE_COLUMNS,
            output_path=artifact_dir / "shap_summary.png",
        )

        report_path = (
            EVALS_DIR / pipeline_id / f"{run_id}.md"
            if pipeline_id
            else EVALS_DIR / f"{run_id}.md"
        )
        write_training_report(
            output_path=report_path,
            model_name=self.model_name,
            run_id=run_id,
            metrics=metrics,
            params=best_params,
            classification_report=report,
            confusion_matrix_markdown=confusion_matrix_to_markdown(matrix),
            shap_image_path=shap_image_path,
            artifact_dir=artifact_dir.relative_to(EVALS_DIR.parent),
        )

        model_path = artifact_dir / "model.joblib"
        tracker.report_metrics("validation", self.model_name, metrics)
        tracker.report_confusion_matrix(
            "validation/confusion_matrix",
            self.model_name,
            matrix,
        )
        tracker.report_table(
            "validation/confusion_matrix_table",
            self.model_name,
            matrix.reset_index(names="actual"),
        )
        tracker.report_text(report)
        self._report_hyperopt_trials(tracker, trials_payload)
        self._report_training_history(tracker, training_history)
        if shap_image_path:
            tracker.report_image(
                "explainability/shap_summary", self.model_name, shap_image_path
            )
        tracker.upload_artifacts(
            {
                "model_joblib": model_path,
                "metrics": artifact_dir / "metrics.json",
                "params": artifact_dir / "params.json",
                "metadata": artifact_dir / "metadata.json",
                "feature_schema": artifact_dir / "feature_schema.json",
                "hyperopt_trials": artifact_dir / "hyperopt_trials.json",
                "training_history": artifact_dir / "training_history.json",
                "train_indices": artifact_dir / "train_indices.json",
                "test_indices": artifact_dir / "test_indices.json",
                "val_indices": artifact_dir / "val_indices.json",
                "split_manifest": artifact_dir / "split_manifest.json",
                "confusion_matrix": artifact_dir / "confusion_matrix.csv",
                "classification_report": artifact_dir / "classification_report.txt",
                "training_report": report_path,
                "shap_summary": shap_image_path or artifact_dir / "missing_shap.png",
            }
        )
        clearml_model_id = tracker.register_output_model(
            model_path=model_path,
            model_name=f"{self.model_name}-{run_id}",
            label_enumeration={
                label: class_id for class_id, label in PRODUCT_CATEGORY_LABELS.items()
            },
            metadata={
                "run_id": run_id,
                "model_name": self.model_name,
                "selection_metric": self.selection_metric,
                "pipeline_id": pipeline_id,
                "clearml_dataset_id": clearml_dataset_id,
                "prediction_endpoint": "/predict",
            },
        )
        if clearml_model_id:
            clearml_model_path = artifact_dir / "clearml_model.json"
            save_json(
                clearml_model_path,
                {
                    "clearml_model_id": clearml_model_id,
                    "clearml_task_id": tracker.task_id,
                    "clearml_dataset_id": clearml_dataset_id,
                },
            )
            tracker.upload_artifact("clearml_model", clearml_model_path)
        if owns_tracker:
            tracker.close()

        logger.success(f"Finished {self.model_name} run {run_id}")
        return TrainingResult(
            model_name=self.model_name,
            run_id=run_id,
            artifact_dir=artifact_dir,
            metrics=metrics,
            params=best_params,
            report_path=report_path,
            clearml_model_id=clearml_model_id,
            clearml_task_id=tracker.task_id,
        )

    def fit_final_pipeline(
        self,
        pipeline: Pipeline | ImbalancedPipeline,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        tracker: ClearMLTracker | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        """Fit the final model and return native per-iteration training history.

        Args:
            pipeline: Unfitted training pipeline.
            x_train: Training features.
            y_train: Training target.
            x_val: Validation features.
            y_val: Validation target.
            tracker: Optional ClearML tracker for live per-iteration logging.

        Returns:
            Nested training history as dataset -> metric -> values.
        """
        _ = x_val, y_val, tracker
        pipeline.fit(x_train, y_train)
        return {}

    def _split_manifest(
        self,
        dataset_path: Path,
        train_indices: list[Any],
        test_indices: list[Any],
    ) -> dict[str, Any]:
        """Build the train/test split manifest for a run.

        Args:
            dataset_path: Source dataset path.
            train_indices: Row indices used for training.
            test_indices: Row indices used for held-out evaluation.

        Returns:
            Split manifest payload.
        """
        return {
            "dataset_path": str(dataset_path),
            "random_state": self.random_state,
            "test_size": DEFAULT_TEST_SIZE,
            "target_column": TARGET_COLUMN,
            "feature_columns": FEATURE_COLUMNS,
            "splits": {
                "train": {
                    "indices_path": "train_indices.json",
                    "row_count": len(train_indices),
                },
                "test": {
                    "indices_path": "test_indices.json",
                    "row_count": len(test_indices),
                    "legacy_alias": "validation",
                    "legacy_indices_path": "val_indices.json",
                },
            },
        }

    def _run_hyperopt(
        self,
        dataframe: pd.DataFrame,
        max_evals: int,
        cv_folds: int,
        use_smote: bool,
        smote_k_neighbors: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run Hyperopt with stratified cross-validation.

        Args:
            dataframe: Training dataframe.
            max_evals: Number of Hyperopt evaluations.
            cv_folds: Number of CV folds.
            use_smote: Whether to use SMOTE in CV folds.
            smote_k_neighbors: SMOTE neighbor count.

        Returns:
            Best parameter dictionary and trial summary payload.
        """
        features, target = split_features_target(dataframe)
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        trials_payload: list[dict[str, Any]] = []

        def objective(raw_params: dict[str, Any]) -> dict[str, Any]:
            params = self.normalize_params(raw_params)
            fold_metrics: list[dict[str, float]] = []
            for train_idx, val_idx in splitter.split(features, target):
                x_train = features.iloc[train_idx]
                y_train = target.iloc[train_idx]
                x_val = features.iloc[val_idx]
                y_val = target.iloc[val_idx]
                pipeline = self.build_pipeline(
                    params,
                    use_smote=use_smote,
                    smote_k_neighbors=smote_k_neighbors,
                )
                pipeline.fit(x_train, y_train)
                metrics, _, _ = evaluate_classifier(pipeline, x_val, y_val)
                fold_metrics.append(metrics)
            mean_metrics = {
                metric: float(np.mean([fold[metric] for fold in fold_metrics]))
                for metric in fold_metrics[0]
            }
            loss = self._metric_to_loss(mean_metrics)
            trials_payload.append(
                {
                    "params": params,
                    "metrics": mean_metrics,
                    "loss": loss,
                }
            )
            return {"loss": loss, "status": STATUS_OK, "metrics": mean_metrics}

        trials = Trials()
        best_raw = fmin(
            fn=objective,
            space=self.hyperopt_space(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
            show_progressbar=True,
        )
        best_params = self.normalize_params(space_eval(self.hyperopt_space(), best_raw))
        logger.info(f"Best params for {self.model_name}: {best_params}")
        return best_params, trials_payload

    def _metric_to_loss(self, metrics: dict[str, float]) -> float:
        """Convert selected metric into a minimization loss.

        Args:
            metrics: Metric dictionary.

        Returns:
            Loss value.
        """
        if self.selection_metric == "log_loss" and "log_loss" in metrics:
            return float(metrics["log_loss"])
        if self.selection_metric in metrics:
            return float(1.0 - metrics[self.selection_metric])
        return float(1.0 - metrics["f1_macro"])

    def _report_hyperopt_trials(
        self,
        tracker: ClearMLTracker,
        trials_payload: list[dict[str, Any]],
    ) -> None:
        """Report hyperparameter search diagnostics to ClearML.

        Args:
            tracker: Active ClearML tracker.
            trials_payload: Hyperopt trial summaries.
        """
        if not trials_payload:
            return
        trials_table = self._trials_to_dataframe(trials_payload)
        tracker.report_table("hyperopt/trials", self.model_name, trials_table)
        tracker.report_scalar_series(
            "hyperopt_cv/loss",
            self.model_name,
            [float(trial["loss"]) for trial in trials_payload],
        )
        best_losses: list[float] = []
        current_best = float("inf")
        for trial in trials_payload:
            current_best = min(current_best, float(trial["loss"]))
            best_losses.append(current_best)
        tracker.report_scalar_series(
            "hyperopt_cv/best_loss_so_far",
            self.model_name,
            best_losses,
        )
        metric_names = sorted(
            {
                metric_name
                for trial in trials_payload
                for metric_name in trial.get("metrics", {}).keys()
            }
        )
        for metric_name in metric_names:
            tracker.report_scalar_series(
                f"hyperopt_cv/{metric_name}",
                self.model_name,
                [
                    float(trial["metrics"][metric_name])
                    for trial in trials_payload
                    if metric_name in trial.get("metrics", {})
                ],
            )

    def _report_training_history(
        self,
        tracker: ClearMLTracker,
        training_history: dict[str, dict[str, list[float]]],
    ) -> None:
        """Report native estimator training curves to ClearML.

        Args:
            tracker: Active ClearML tracker.
            training_history: Nested dataset -> metric -> values payload.
        """
        if not training_history:
            return
        tracker.report_table(
            "training_iterations/history",
            self.model_name,
            self._training_history_to_dataframe(training_history),
        )
        for dataset_name, metrics in training_history.items():
            for metric_name, values in metrics.items():
                if not values:
                    continue
                tracker.report_scalar_series(
                    f"training_iterations/{metric_name}",
                    f"{self.model_name}/{dataset_name}",
                    [float(value) for value in values],
                )

    @staticmethod
    def _training_history_to_dataframe(
        training_history: dict[str, dict[str, list[float]]],
    ) -> pd.DataFrame:
        """Convert native estimator training history into a long dataframe.

        Args:
            training_history: Nested dataset -> metric -> values payload.

        Returns:
            Long dataframe with iteration, dataset, metric, and value columns.
        """
        rows: list[dict[str, Any]] = []
        for dataset_name, metrics in training_history.items():
            for metric_name, values in metrics.items():
                for iteration, value in enumerate(values):
                    rows.append(
                        {
                            "iteration": iteration,
                            "dataset": dataset_name,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )
        return pd.DataFrame(rows)

    def _trials_to_dataframe(
        self, trials_payload: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Convert Hyperopt trials into a flat table.

        Args:
            trials_payload: Hyperopt trial summaries.

        Returns:
            Flat dataframe with params, metrics, and loss.
        """
        rows: list[dict[str, Any]] = []
        for iteration, trial in enumerate(trials_payload):
            row: dict[str, Any] = {
                "iteration": iteration,
                "loss": trial.get("loss"),
            }
            for metric_name, value in trial.get("metrics", {}).items():
                row[f"metric_{metric_name}"] = value
            for param_name, value in trial.get("params", {}).items():
                row[f"param_{param_name}"] = value
            rows.append(row)
        return pd.DataFrame(rows)
