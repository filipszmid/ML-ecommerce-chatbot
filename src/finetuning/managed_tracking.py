"""ClearML tracking helpers for managed-provider fine-tuning jobs."""

# Provider tracking records explicit job metadata fields for stable manifests.

import time
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import CLEARML_SERVING_BASE_URL, CLEARML_SERVING_ENDPOINT
from src.models.common.artifacts import save_json


def start_managed_finetune_tracking(
    provider_name: str,
    dataset_path: str,
    run_name: str | None,
    params: dict[str, Any],
    clearml_enabled: bool,
) -> tuple[ClearMLTracker, Path, str | None]:
    """Start ClearML tracking for a managed-provider fine-tuning submission.

    Args:
        provider_name: Provider key.
        dataset_path: Local path, cloud URI, or provider file id.
        run_name: Optional user-facing run name.
        params: Provider-specific job parameters.
        clearml_enabled: Whether tracking is enabled.

    Returns:
        Tracker, local run directory, and ClearML dataset id if one was registered.
    """
    resolved_run_name = run_name or f"{int(time.time())}_{provider_name}_finetune"
    run_dir = Path("data/runs/llm_finetuning") / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = ClearMLTracker(enabled=clearml_enabled)
    tracker.start(
        task_name=resolved_run_name,
        params={
            "provider": provider_name,
            "dataset_path": dataset_path,
            **params,
        },
        task_type="training",
        tags=["llm", "finetuning", provider_name, "managed-provider"],
    )

    dataset_id = _log_dataset_or_manifest(
        tracker=tracker,
        provider_name=provider_name,
        dataset_path=dataset_path,
        run_name=resolved_run_name,
        run_dir=run_dir,
    )
    return tracker, run_dir, dataset_id


def finish_managed_finetune_tracking(
    tracker: ClearMLTracker,
    run_dir: Path,
    provider_name: str,
    dataset_path: str,
    dataset_id: str | None,
    model_id: str,
    job_id: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    """Log the submitted managed-provider fine-tuning job into ClearML.

    Args:
        tracker: Active ClearML tracker.
        run_dir: Local run directory.
        provider_name: Provider key.
        dataset_path: Local path, cloud URI, or provider file id.
        dataset_id: ClearML dataset id, if registered.
        model_id: Provider base/source model id.
        job_id: Provider job id or ARN.
        request_payload: Submitted request payload without secrets.
        response_payload: Provider response payload without secrets.

    Returns:
        Fine-tuning manifest.
    """
    serving_endpoint = (
        f"{CLEARML_SERVING_BASE_URL.rstrip('/')}/"
        f"{CLEARML_SERVING_ENDPOINT.strip('/')}-{provider_name}"
    )
    manifest = {
        "provider": provider_name,
        "dataset_path": dataset_path,
        "clearml_dataset_id": dataset_id,
        "model_id": model_id,
        "job_id": job_id,
        "request_payload": request_payload,
        "response_payload": response_payload,
        "clearml_managed_serving_endpoint": serving_endpoint,
        "status": "submitted",
    }
    manifest_path = run_dir / "managed_fine_tune_manifest.json"
    save_json(manifest_path, manifest)
    save_json(Path("data/runs/llm_finetuning/latest_managed_finetune.json"), manifest)

    tracker.report_table(
        "managed_finetuning/job",
        provider_name,
        pd.DataFrame(
            [
                {
                    "provider": provider_name,
                    "job_id": job_id,
                    "model_id": model_id,
                    "dataset_path": dataset_path,
                    "status": "submitted",
                }
            ]
        ),
    )
    tracker.upload_artifact("managed_fine_tune_manifest", manifest_path)
    clearml_model_id = tracker.register_output_model(
        model_path=manifest_path,
        model_name=f"remote-llm-finetune-{provider_name}-{job_id}".replace("/", "-"),
        metadata=manifest,
        framework=f"remote-{provider_name}",
    )
    if clearml_model_id:
        manifest["clearml_model_id"] = clearml_model_id
        save_json(manifest_path, manifest)
        save_json(
            Path("data/runs/llm_finetuning/latest_managed_finetune.json"),
            manifest,
        )
        tracker.upload_artifact(
            "managed_fine_tune_manifest_with_model_id", manifest_path
        )
    tracker.close()
    return manifest


def log_managed_finetune_error(
    tracker: ClearMLTracker,
    run_dir: Path,
    provider_name: str,
    dataset_path: str,
    model_id: str,
    request_payload: dict[str, Any],
    error: Exception,
) -> None:
    """Log a failed managed-provider submission before re-raising.

    Args:
        tracker: Active ClearML tracker.
        run_dir: Local run directory.
        provider_name: Provider key.
        dataset_path: Local path, cloud URI, or provider file id.
        model_id: Provider base/source model id.
        request_payload: Submitted request payload without secrets.
        error: Submission error.
    """
    manifest_path = run_dir / "managed_fine_tune_error.json"
    save_json(
        manifest_path,
        {
            "provider": provider_name,
            "dataset_path": dataset_path,
            "model_id": model_id,
            "request_payload": request_payload,
            "error": str(error),
            "status": "failed_to_submit",
        },
    )
    tracker.upload_artifact("managed_fine_tune_error", manifest_path)
    tracker.report_text(f"{provider_name} fine-tuning submission failed: {error}")
    tracker.close()


def _log_dataset_or_manifest(
    tracker: ClearMLTracker,
    provider_name: str,
    dataset_path: str,
    run_name: str,
    run_dir: Path,
) -> str | None:
    """Log a local dataset or a remote dataset manifest.

    Args:
        tracker: Active ClearML tracker.
        provider_name: Provider key.
        dataset_path: Local path, cloud URI, or provider file id.
        run_name: Dataset version.
        run_dir: Local run directory.

    Returns:
        ClearML dataset id for local datasets, otherwise None.
    """
    local_dataset_path = Path(dataset_path)
    if local_dataset_path.exists():
        return tracker.log_dataset(
            dataset_path=local_dataset_path,
            dataset_name=f"llm_{provider_name}_finetune_dataset",
            dataset_version=run_name,
            tags=["llm", "finetuning", provider_name],
        )

    manifest_path = run_dir / "dataset_manifest.json"
    save_json(
        manifest_path,
        {
            "provider": provider_name,
            "dataset_path": dataset_path,
            "dataset_reference_type": _dataset_reference_type(dataset_path),
        },
    )
    tracker.upload_artifact("dataset_manifest", manifest_path)
    tracker.report_table(
        "managed_finetuning/dataset",
        provider_name,
        pd.DataFrame(
            [
                {
                    "provider": provider_name,
                    "dataset_path": dataset_path,
                    "dataset_reference_type": _dataset_reference_type(dataset_path),
                }
            ]
        ),
    )
    return None


def _dataset_reference_type(dataset_path: str) -> str:
    """Classify a managed-provider dataset reference.

    Args:
        dataset_path: Dataset path or provider id.

    Returns:
        Dataset reference type.
    """
    if dataset_path.startswith("file-"):
        return "azure_openai_file_id"
    if dataset_path.startswith("s3://"):
        return "s3_uri"
    if dataset_path.startswith("gs://"):
        return "gcs_uri"
    return "remote_or_provider_reference"
