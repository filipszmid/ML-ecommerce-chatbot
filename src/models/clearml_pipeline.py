"""ClearML Pipeline execution for model selection."""

# ClearML PipelineDecorator components expose explicit parameters to the
# pipeline UI, so their signatures stay stable. Candidate failures are captured
# and reported by the controller when continue-on-error is enabled.

import csv
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from clearml import PipelineDecorator

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import (
    CLASSICAL_RUNS_DIR,
    CLEARML_FILES_HOST,
    CLEARML_PROJECT_NAME,
    CLEARML_SERVING_BASE_URL,
    CLEARML_SERVING_ENDPOINT,
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
from src.models.registry import get_workflow, parse_model_names


def run_clearml_model_selection_pipeline(
    models: str | None = "all",
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    max_evals: int = DEFAULT_MAX_EVALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    use_smote: bool = DEFAULT_USE_SMOTE,
    smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
    selection_metric: str = MODEL_SELECTION_METRIC,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Run model selection as a local ClearML Pipeline.

    Args:
        models: Comma-separated model names or "all".
        dataset_path: Training CSV path.
        max_evals: Hyperopt evaluations per model.
        cv_folds: Stratified CV folds.
        use_smote: Whether to use SMOTE during training and CV.
        smote_k_neighbors: SMOTE neighbor count.
        selection_metric: Metric used to rank models.
        continue_on_error: Whether to keep training after a model failure.

    Returns:
        Selection summary payload.
    """
    PipelineDecorator.run_locally()
    return _model_selection_pipeline(
        models=models,
        dataset_path=str(dataset_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        selection_metric=selection_metric,
        continue_on_error=continue_on_error,
    )


@PipelineDecorator.pipeline(
    name="model-selection",
    project=CLEARML_PROJECT_NAME,
    version="1.0",
    add_pipeline_tags=True,
    target_project=True,
    start_controller_locally=True,
    pipeline_execution_queue=None,
    output_uri=CLEARML_FILES_HOST,
)
def _model_selection_pipeline(
    models: str | None,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    selection_metric: str,
    continue_on_error: bool,
) -> dict[str, Any]:
    """Build the ClearML pipeline graph for model selection."""
    ensure_artifact_dirs()
    selection_id = f"{utc_timestamp()}_model_selection"
    model_names = parse_model_names(models)
    dataset_id = _register_dataset(selection_id, dataset_path)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for model_name in model_names:
        try:
            result = _train_model_step(
                model_name=model_name,
                selection_id=selection_id,
                dataset_path=dataset_path,
                max_evals=max_evals,
                cv_folds=cv_folds,
                use_smote=use_smote,
                smote_k_neighbors=smote_k_neighbors,
                clearml_dataset_id=dataset_id,
            )
            results.append(result)
            _report_controller_model_result(
                model_name=model_name,
                iteration=len(results),
                result=result,
            )
        except Exception as exc:
            failures.append({"model_name": model_name, "error": str(exc)})
            _report_controller_failure(
                model_name=model_name,
                iteration=len(results) + len(failures),
                error=str(exc),
            )
            if not continue_on_error:
                raise

    return _finalize_model_selection(
        selection_id=selection_id,
        selection_metric=selection_metric,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_dataset_id=dataset_id,
        results=results,
        failures=failures,
    )


def _report_controller_model_result(
    model_name: str,
    iteration: int,
    result: dict[str, Any],
) -> None:
    """Report model-selection progress to the pipeline controller task.

    Args:
        model_name: Candidate model name.
        iteration: Pipeline progress iteration.
        result: Candidate training result payload.
    """
    tracker = ClearMLTracker(enabled=True)
    tracker.start(
        task_name="model-selection-controller",
        params={"controller_metric_source": "model_selection"},
        task_type="controller",
        tags=["classical-ml", "model-selection", "controller"],
    )
    for metric_name, value in result.get("metrics", {}).items():
        scalar_value = _as_float(value)
        if scalar_value is None:
            continue
        tracker.report_scalar_points(
            f"pipeline_model_selection/{metric_name}",
            model_name,
            [(iteration, scalar_value)],
        )
    tracker.report_scalar_points(
        "pipeline_model_selection/progress",
        "completed_models",
        [(iteration, float(iteration))],
    )


def _report_controller_failure(
    model_name: str,
    iteration: int,
    error: str,
) -> None:
    """Report a model-selection failure to the pipeline controller task.

    Args:
        model_name: Candidate model name.
        iteration: Pipeline progress iteration.
        error: Failure message.
    """
    tracker = ClearMLTracker(enabled=True)
    tracker.start(
        task_name="model-selection-controller",
        params={"controller_metric_source": "model_selection"},
        task_type="controller",
        tags=["classical-ml", "model-selection", "controller"],
    )
    tracker.report_scalar_points(
        "pipeline_model_selection/failures",
        model_name,
        [(iteration, 1.0)],
    )
    tracker.report_text(f"Model-selection candidate failed: {model_name}: {error}")


def _as_float(value: Any) -> float | None:
    """Convert numeric values to floats and ignore non-scalars.

    Args:
        value: Candidate scalar value.

    Returns:
        Float value, if numeric.
    """
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


@PipelineDecorator.component(
    name="register_dataset",
    return_values=["dataset_id"],
    task_type="data_processing",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _register_dataset(selection_id: str, dataset_path: str) -> str | None:
    """Register the training CSV as a ClearML Dataset."""
    tracker = ClearMLTracker(enabled=True)
    tracker.start(
        task_name=f"{selection_id}_dataset",
        params={"selection_id": selection_id, "dataset_path": dataset_path},
        task_type="data_processing",
        tags=["classical-ml", "dataset", "model-selection"],
    )
    dataset_id = tracker.log_dataset(
        dataset_path=Path(dataset_path),
        dataset_name="customer_purchase_data",
        dataset_version=selection_id,
        tags=["tabular", "ecommerce", "model-selection"],
    )
    # Do not call tracker.close() — the pipeline controller manages the task lifecycle.
    return dataset_id


def _train_model_step(
    model_name: str,
    selection_id: str,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
) -> dict[str, Any]:
    """Dispatch candidate training to a named ClearML Pipeline step."""
    train_step = {
        "xgboost": _train_xgboost,
        "randomforest": _train_randomforest,
        "catboost": _train_catboost,
        "logistic_regression": _train_logistic_regression,
        "logreg": _train_logistic_regression,
        "lda": _train_lda,
    }[model_name]
    return train_step(
        selection_id=selection_id,
        dataset_path=dataset_path,
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_dataset_id=clearml_dataset_id,
    )


@PipelineDecorator.component(
    name="train_xgboost",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _train_xgboost(
    selection_id: str,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
) -> dict[str, Any]:
    """Train the XGBoost candidate."""
    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    workflow = get_workflow("xgboost")
    result = workflow.run(
        dataset_path=Path(dataset_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=True,
        pipeline_id=selection_id,
        clearml_dataset_id=clearml_dataset_id,
    )
    payload = asdict(result)
    payload["artifact_dir"] = str(result.artifact_dir.relative_to(DATA_DIR))
    payload["report_path"] = str(result.report_path.relative_to(DATA_DIR))
    return payload


@PipelineDecorator.component(
    name="train_randomforest",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _train_randomforest(
    selection_id: str,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
) -> dict[str, Any]:
    """Train the Random Forest candidate."""
    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    workflow = get_workflow("randomforest")
    result = workflow.run(
        dataset_path=Path(dataset_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=True,
        pipeline_id=selection_id,
        clearml_dataset_id=clearml_dataset_id,
    )
    payload = asdict(result)
    payload["artifact_dir"] = str(result.artifact_dir.relative_to(DATA_DIR))
    payload["report_path"] = str(result.report_path.relative_to(DATA_DIR))
    return payload


@PipelineDecorator.component(
    name="train_catboost",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _train_catboost(
    selection_id: str,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
) -> dict[str, Any]:
    """Train the CatBoost candidate."""
    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    workflow = get_workflow("catboost")
    result = workflow.run(
        dataset_path=Path(dataset_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=True,
        pipeline_id=selection_id,
        clearml_dataset_id=clearml_dataset_id,
    )
    payload = asdict(result)
    payload["artifact_dir"] = str(result.artifact_dir.relative_to(DATA_DIR))
    payload["report_path"] = str(result.report_path.relative_to(DATA_DIR))
    return payload


@PipelineDecorator.component(
    name="train_logistic_regression",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _train_logistic_regression(
    selection_id: str,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
) -> dict[str, Any]:
    """Train the Logistic Regression candidate."""
    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    workflow = get_workflow("logistic_regression")
    result = workflow.run(
        dataset_path=Path(dataset_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=True,
        pipeline_id=selection_id,
        clearml_dataset_id=clearml_dataset_id,
    )
    payload = asdict(result)
    payload["artifact_dir"] = str(result.artifact_dir.relative_to(DATA_DIR))
    payload["report_path"] = str(result.report_path.relative_to(DATA_DIR))
    return payload


@PipelineDecorator.component(
    name="train_lda",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _train_lda(
    selection_id: str,
    dataset_path: str,
    max_evals: int,
    cv_folds: int,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
) -> dict[str, Any]:
    """Train the LDA candidate."""
    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    workflow = get_workflow("lda")
    result = workflow.run(
        dataset_path=Path(dataset_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=True,
        pipeline_id=selection_id,
        clearml_dataset_id=clearml_dataset_id,
    )
    payload = asdict(result)
    payload["artifact_dir"] = str(result.artifact_dir.relative_to(DATA_DIR))
    payload["report_path"] = str(result.report_path.relative_to(DATA_DIR))
    return payload


@PipelineDecorator.component(
    name="finalize_model_selection",
    return_values=["payload"],
    task_type="data_processing",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _finalize_model_selection(
    selection_id: str,
    selection_metric: str,
    use_smote: bool,
    smote_k_neighbors: int,
    clearml_dataset_id: str | None,
    results: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    """Rank candidate models and log the selection summary."""
    if not results:
        raise RuntimeError("No model-selection results were produced")

    def _ranking_value(metrics: dict[str, float]) -> float:
        if selection_metric in metrics:
            return metrics[selection_metric]
        return metrics["f1_macro"]

    ranked = sorted(
        results,
        key=lambda result: _ranking_value(result["metrics"]),
        reverse=selection_metric != "log_loss",
    )
    best = ranked[0]

    # Build deployment manifest.
    serving_base_url = CLEARML_SERVING_BASE_URL.rstrip("/")
    serving_endpoint_path = CLEARML_SERVING_ENDPOINT.strip("/")
    serving_endpoint = f"{serving_base_url}/{serving_endpoint_path}"

    deployment_manifest = {
        "selection_id": selection_id,
        "selected_model": best["model_name"],
        "selected_run_id": best["run_id"],
        "selected_artifact_dir": best["artifact_dir"],
        "selected_model_path": f"{best['artifact_dir']}/model.joblib",
        "selected_metadata_path": f"{best['artifact_dir']}/metadata.json",
        "local_prediction_endpoint": "http://localhost:8000/predict",
        "local_chat_endpoint": "http://localhost:8001",
        "clearml_serving_base_url": serving_base_url,
        "clearml_serving_endpoint": serving_endpoint,
        "docker_prediction_endpoint": "http://api:8000/predict",
        "docker_chat_endpoint": "http://chat:8001",
        "cloud_endpoint_targets": {
            "aws": "SageMaker endpoint created by infra/aws",
            "azure": "Azure ML online endpoint created by infra/azure",
            "gcp": "Vertex AI endpoint created by infra/gcp",
        },
    }

    # Build endpoint manifest.
    endpoint_manifest = {
        "local": {
            "chat_ui": "http://localhost:8001",
            "prediction_api": "http://localhost:8000",
            "prediction_endpoint": "http://localhost:8000/predict",
            "model_status_endpoint": "http://localhost:8000/models/latest",
        },
        "clearml_serving": {
            "base": serving_base_url,
            "product_category_endpoint": serving_endpoint,
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

    payload = {
        "selection_id": selection_id,
        "selection_metric": selection_metric,
        "use_smote": use_smote,
        "smote_k_neighbors": smote_k_neighbors,
        "clearml_dataset_id": clearml_dataset_id,
        "best_model": best["model_name"],
        "best_run_id": best["run_id"],
        "best_artifact_dir": best["artifact_dir"],
        "deployment_manifest": deployment_manifest,
        "runs": ranked,
        "failures": failures,
    }

    selection_dir = EVALS_DIR / selection_id
    save_json(selection_dir / "selection.json", payload)
    save_json(selection_dir / "deployment_manifest.json", deployment_manifest)
    save_json(selection_dir / "endpoint_manifest.json", endpoint_manifest)

    # Write selection CSV.
    metric_names = sorted({key for result in ranked for key in result["metrics"]})
    csv_path = selection_dir / "selection.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["rank", "model_name", "run_id", *metric_names, "artifact_dir"],
        )
        writer.writeheader()
        for rank, result in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "model_name": result["model_name"],
                    "run_id": result["run_id"],
                    "artifact_dir": result["artifact_dir"],
                    **result["metrics"],
                }
            )

    # Write selection Markdown.
    md_path = selection_dir / "selection.md"
    header = "| Rank | Model | Run | " + " | ".join(metric_names) + " |"
    separator = (
        "| ---: | --- | --- | " + " | ".join(["---:"] * len(metric_names)) + " |"
    )
    md_rows = []
    for rank, result in enumerate(ranked, start=1):
        metric_cells = [
            f"{result['metrics'].get(metric, 0.0):.6f}" for metric in metric_names
        ]
        md_rows.append(
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
    md_path.write_text(
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
                *md_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )

    save_json(CLASSICAL_RUNS_DIR / "latest_selection.json", payload)
    save_json(
        CLASSICAL_RUNS_DIR / "latest_deployment_manifest.json", deployment_manifest
    )
    save_json(CLASSICAL_RUNS_DIR / "latest_endpoint_manifest.json", endpoint_manifest)

    # Log to ClearML.
    tracker = ClearMLTracker(enabled=True)
    tracker.start(
        task_name=f"{payload['selection_id']}_finalize",
        params={"selection_id": payload["selection_id"]},
        task_type="data_processing",
        tags=["classical-ml", "model-selection", "finalize"],
    )
    table = pd.DataFrame(
        [
            {
                "rank": rank,
                "model_name": result["model_name"],
                "run_id": result["run_id"],
                "artifact_dir": result["artifact_dir"],
                "clearml_model_id": result.get("clearml_model_id"),
                **result["metrics"],
            }
            for rank, result in enumerate(ranked, start=1)
        ]
    )
    endpoint_rows = []
    for scope, endpoints in endpoint_manifest.items():
        for name, value in endpoints.items():
            endpoint_rows.append({"scope": scope, "name": name, "value": value})
    endpoint_df = pd.DataFrame(endpoint_rows)

    tracker.report_table("model_selection/ranking", "candidates", table)
    tracker.report_table("serving/endpoints", "local_docker_clearml", endpoint_df)
    tracker.report_metrics(
        "model_selection/best", payload["best_model"], ranked[0]["metrics"]
    )
    for metric_name in metric_names:
        tracker.report_scalar_series(
            f"model_selection/{metric_name}",
            "ranked_candidates",
            [float(result["metrics"][metric_name]) for result in ranked],
        )
    tracker.upload_artifacts(
        {
            "selection_json": selection_dir / "selection.json",
            "selection_csv": selection_dir / "selection.csv",
            "selection_report": selection_dir / "selection.md",
            "deployment_manifest": selection_dir / "deployment_manifest.json",
            "endpoint_manifest": selection_dir / "endpoint_manifest.json",
            "best_model_joblib": DATA_DIR / ranked[0]["artifact_dir"] / "model.joblib",
            "best_model_metadata": DATA_DIR
            / ranked[0]["artifact_dir"]
            / "metadata.json",
            "best_model_metrics": DATA_DIR / ranked[0]["artifact_dir"] / "metrics.json",
        }
    )
    # Do not call tracker.close() — the pipeline controller manages the task lifecycle.
    return payload
