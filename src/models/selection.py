"""Model selection workflow across classical algorithms."""

# The workflow is an orchestration boundary with stable CLI/ClearML parameters.
# ClearML and joblib backend imports stay lazy to avoid changing normal startup.

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import (
    ADK_WEB_URL,
    API_PORT,
    CLASSICAL_RUNS_DIR,
    DEFAULT_CV_FOLDS,
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_EVALS,
    DEFAULT_SMOTE_K_NEIGHBORS,
    DEFAULT_USE_SMOTE,
    DATA_DIR,
    EVALS_DIR,
    MODEL_SELECTION_METRIC,
)
from src.models.common.artifacts import ensure_artifact_dirs, save_json, utc_timestamp
from src.models.common.base import TrainingResult
from src.models.registry import get_workflow, parse_model_names


class ModelSelectionWorkflow:
    """Run model selection across registered workflows."""

    def __init__(self, selection_metric: str = MODEL_SELECTION_METRIC) -> None:
        """Initialize workflow.

        Args:
            selection_metric: Metric used to rank models.
        """
        self.selection_metric = selection_metric

    def run(
        self,
        models: str | None = "all",
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        max_evals: int = DEFAULT_MAX_EVALS,
        cv_folds: int = DEFAULT_CV_FOLDS,
        use_smote: bool = DEFAULT_USE_SMOTE,
        smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
        clearml_enabled: bool = False,
        continue_on_error: bool = False,
    ) -> dict[str, Any]:
        """Train candidate models and write a selection report.

        Args:
            models: Comma-separated model names or "all".
            dataset_path: Training CSV path.
            max_evals: Hyperopt evaluations per model.
            cv_folds: Stratified CV folds.
            use_smote: Whether to use SMOTE during training and CV.
            smote_k_neighbors: SMOTE neighbor count.
            clearml_enabled: Whether to log runs into ClearML.
            continue_on_error: Whether to keep training after a model failure.

        Returns:
            Selection summary payload.
        """
        if clearml_enabled and os.getenv("CLEARML_PIPELINE_INTERNAL") != "true":
            from src.models.clearml_pipeline import run_clearml_model_selection_pipeline

            return run_clearml_model_selection_pipeline(
                models=models,
                dataset_path=dataset_path,
                max_evals=max_evals,
                cv_folds=cv_folds,
                use_smote=use_smote,
                smote_k_neighbors=smote_k_neighbors,
                selection_metric=self.selection_metric,
                continue_on_error=continue_on_error,
            )

        ensure_artifact_dirs()
        selection_id = f"{utc_timestamp()}_model_selection"
        model_names = parse_model_names(models)
        results: list[TrainingResult] = []
        failures: list[dict[str, str]] = []
        selection_tracker = ClearMLTracker(enabled=clearml_enabled)
        selection_tracker.start(
            task_name=selection_id,
            params={
                "selection_id": selection_id,
                "models": model_names,
                "dataset_path": str(dataset_path),
                "max_evals": max_evals,
                "cv_folds": cv_folds,
                "use_smote": use_smote,
                "smote_k_neighbors": smote_k_neighbors,
                "selection_metric": self.selection_metric,
                "continue_on_error": continue_on_error,
            },
            task_type="controller",
            tags=[
                "classical-ml",
                "pipeline",
                "model-selection",
                "hyperopt",
                f"smote:{str(use_smote).lower()}",
            ],
        )
        clearml_dataset_id = selection_tracker.log_dataset(
            dataset_path=Path(dataset_path),
            dataset_name="customer_purchase_data",
            dataset_version=selection_id,
            tags=["tabular", "ecommerce", "model-selection"],
        )

        try:
            import joblib

            # Enforce the threading backend globally for all Scikit-Learn / joblib
            # tasks. The loky process backend is unstable when mixed with
            # CatBoost/XGBoost native libs.
            with joblib.parallel_backend("threading", n_jobs=-1):
                for model_name in model_names:
                    workflow = get_workflow(model_name)
                    try:
                        result = workflow.run(
                            dataset_path=dataset_path,
                            max_evals=max_evals,
                            cv_folds=cv_folds,
                            use_smote=use_smote,
                            smote_k_neighbors=smote_k_neighbors,
                            clearml_enabled=clearml_enabled,
                            pipeline_id=selection_id,
                            clearml_dataset_id=clearml_dataset_id,
                            clearml_tracker=selection_tracker,
                        )
                        results.append(result)
                    except Exception as exc:
                        logger.exception(f"Model {model_name} failed")
                        failures.append({"model_name": model_name, "error": str(exc)})
                        if not continue_on_error:
                            raise

            if not results:
                raise RuntimeError("No model-selection results were produced")

            ranked = sorted(
                results,
                key=lambda result: self._ranking_value(result.metrics),
                reverse=self.selection_metric != "log_loss",
            )
            best = ranked[0]
            deployment_manifest = self._deployment_manifest(selection_id, best)
            payload = {
                "selection_id": selection_id,
                "selection_metric": self.selection_metric,
                "use_smote": use_smote,
                "smote_k_neighbors": smote_k_neighbors,
                "clearml_dataset_id": clearml_dataset_id,
                "best_model": best.model_name,
                "best_run_id": best.run_id,
                "best_artifact_dir": str(best.artifact_dir.relative_to(DATA_DIR)),
                "deployment_manifest": deployment_manifest,
                "runs": [
                    {
                        **asdict(result),
                        "artifact_dir": str(result.artifact_dir.relative_to(DATA_DIR)),
                        "report_path": str(result.report_path.relative_to(DATA_DIR)),
                    }
                    for result in ranked
                ],
                "failures": failures,
            }
            selection_dir = EVALS_DIR / selection_id
            save_json(selection_dir / "selection.json", payload)
            save_json(selection_dir / "deployment_manifest.json", deployment_manifest)
            save_json(
                selection_dir / "endpoint_manifest.json", self._endpoint_manifest()
            )
            self._write_csv(selection_dir / "selection.csv", ranked)
            self._write_markdown(selection_dir / "selection.md", payload)
            save_json(CLASSICAL_RUNS_DIR / "latest_selection.json", payload)
            save_json(
                CLASSICAL_RUNS_DIR / "latest_deployment_manifest.json",
                deployment_manifest,
            )
            save_json(
                CLASSICAL_RUNS_DIR / "latest_endpoint_manifest.json",
                self._endpoint_manifest(),
            )
            self._log_selection_to_clearml(
                tracker=selection_tracker,
                ranked=ranked,
                payload=payload,
                selection_dir=selection_dir,
            )
            return payload
        finally:
            selection_tracker.close()

    def _ranking_value(self, metrics: dict[str, float]) -> float:
        """Read the metric used for ranking.

        Args:
            metrics: Metrics payload.

        Returns:
            Ranking value.
        """
        if self.selection_metric in metrics:
            return metrics[self.selection_metric]
        return metrics["f1_macro"]

    def _deployment_manifest(
        self,
        selection_id: str,
        best: TrainingResult,
    ) -> dict[str, Any]:
        """Build a deployment manifest for the selected model.

        Args:
            selection_id: Model-selection identifier.
            best: Best ranked training result.

        Returns:
            Deployment manifest payload.
        """
        return {
            "selection_id": selection_id,
            "selected_model": best.model_name,
            "selected_run_id": best.run_id,
            "selected_artifact_dir": str(best.artifact_dir.relative_to(DATA_DIR)),
            "selected_model_path": str(
                (best.artifact_dir / "model.joblib").relative_to(DATA_DIR)
            ),
            "selected_metadata_path": str(
                (best.artifact_dir / "metadata.json").relative_to(DATA_DIR)
            ),
            "local_prediction_endpoint": f"http://localhost:{API_PORT}/predict",
            "local_chat_endpoint": ADK_WEB_URL,
            "docker_prediction_endpoint": "http://api:8000/predict",
            "docker_chat_endpoint": "http://chat:8001",
            "cloud_endpoint_targets": {
                "aws": "SageMaker endpoint created by infra/aws",
                "azure": "Azure ML online endpoint created by infra/azure",
                "gcp": "Vertex AI endpoint created by infra/gcp",
            },
        }

    def _endpoint_manifest(self) -> dict[str, Any]:
        """Build endpoint metadata for the demo services.

        Returns:
            Endpoint manifest payload.
        """
        return {
            "local": {
                "chat_ui": ADK_WEB_URL,
                "prediction_api": f"http://localhost:{API_PORT}",
                "prediction_endpoint": f"http://localhost:{API_PORT}/predict",
                "model_status_endpoint": f"http://localhost:{API_PORT}/models/latest",
            },
            "docker": {
                "chat_ui": "http://localhost:8001",
                "prediction_api": "http://localhost:8000",
                "internal_prediction_endpoint": "http://api:8000/predict",
            },
            "cloud": {
                "aws": "SageMaker endpoint proxied by Lambda",
                "azure": "Azure ML online endpoint",
                "gcp": "Vertex AI endpoint proxied by Cloud Function",
            },
        }

    def _log_selection_to_clearml(
        self,
        tracker: ClearMLTracker,
        ranked: list[TrainingResult],
        payload: dict[str, Any],
        selection_dir: Path,
    ) -> None:
        """Log model-selection outputs to ClearML.

        Args:
            tracker: Active ClearML tracker.
            ranked: Ranked training results.
            payload: Selection payload.
            selection_dir: Local selection output directory.
        """
        if not ranked:
            return
        table = self._selection_dataframe(ranked)
        tracker.report_table("model_selection/ranking", "candidates", table)
        tracker.report_table(
            "serving/endpoints",
            "local_and_docker",
            self._endpoint_dataframe(),
        )
        tracker.report_metrics(
            "model_selection/best", payload["best_model"], ranked[0].metrics
        )
        metric_names = sorted({key for result in ranked for key in result.metrics})
        for metric_name in metric_names:
            tracker.report_scalar_series(
                f"model_selection/{metric_name}",
                "ranked_candidates",
                [
                    float(result.metrics[metric_name])
                    for result in ranked
                    if metric_name in result.metrics
                ],
            )
        tracker.report_text(
            f"Best model: {payload['best_model']} ({payload['best_run_id']})"
        )
        tracker.upload_artifacts(
            {
                "selection_json": selection_dir / "selection.json",
                "selection_csv": selection_dir / "selection.csv",
                "selection_report": selection_dir / "selection.md",
                "deployment_manifest": selection_dir / "deployment_manifest.json",
                "endpoint_manifest": selection_dir / "endpoint_manifest.json",
                "best_model_joblib": ranked[0].artifact_dir / "model.joblib",
                "best_model_metadata": ranked[0].artifact_dir / "metadata.json",
                "best_model_metrics": ranked[0].artifact_dir / "metrics.json",
            }
        )

    def _selection_dataframe(self, results: list[TrainingResult]) -> pd.DataFrame:
        """Build a ranked model-selection dataframe.

        Args:
            results: Ranked training results.

        Returns:
            Selection dataframe.
        """
        rows = []
        for rank, result in enumerate(results, start=1):
            rows.append(
                {
                    "rank": rank,
                    "model_name": result.model_name,
                    "run_id": result.run_id,
                    "artifact_dir": str(result.artifact_dir),
                    **result.metrics,
                }
            )
        return pd.DataFrame(rows)

    def _endpoint_dataframe(self) -> pd.DataFrame:
        """Build a dataframe with serving endpoints.

        Returns:
            Endpoint dataframe.
        """
        manifest = self._endpoint_manifest()
        rows = []
        for scope, endpoints in manifest.items():
            for name, url in endpoints.items():
                rows.append({"scope": scope, "name": name, "value": url})
        return pd.DataFrame(rows)

    def _write_csv(self, path: Path, results: list[TrainingResult]) -> None:
        """Write selection summary CSV.

        Args:
            path: Output path.
            results: Ranked results.
        """
        metric_names = sorted({key for result in results for key in result.metrics})
        with path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "rank",
                    "model_name",
                    "run_id",
                    *metric_names,
                    "artifact_dir",
                ],
            )
            writer.writeheader()
            for rank, result in enumerate(results, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "model_name": result.model_name,
                        "run_id": result.run_id,
                        "artifact_dir": result.artifact_dir,
                        **result.metrics,
                    }
                )

    def _write_markdown(self, path: Path, payload: dict[str, Any]) -> None:
        """Write selection summary Markdown.

        Args:
            path: Output path.
            payload: Selection payload.
        """
        metric_names = sorted(
            {key for result in payload["runs"] for key in result["metrics"].keys()}
        )
        header = "| Rank | Model | Run | " + " | ".join(metric_names) + " |"
        separator = (
            "| ---: | --- | --- | " + " | ".join(["---:"] * len(metric_names)) + " |"
        )
        rows = []
        for rank, result in enumerate(payload["runs"], start=1):
            metric_cells = [
                f"{result['metrics'].get(metric, 0.0):.6f}" for metric in metric_names
            ]
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        result["model_name"],
                        f"`{result['run_id']}`",
                        *metric_cells,
                    ]
                )
                + " |"
            )
        path.write_text(
            "\n".join(
                [
                    "# Model Selection",
                    "",
                    f"- Selection id: `{payload['selection_id']}`",
                    f"- Selection metric: `{payload['selection_metric']}`",
                    f"- Best model: `{payload['best_model']}`",
                    f"- Best run: `{payload['best_run_id']}`",
                    "",
                    header,
                    separator,
                    *rows,
                    "",
                ]
            )
        )
