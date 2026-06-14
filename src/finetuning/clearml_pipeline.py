"""ClearML Pipeline wrapper for local Ollama fine-tuning."""

# ClearML PipelineDecorator components expose explicit parameters to the
# pipeline UI; keeping those signatures stable is part of the runtime contract.
# Component imports are delayed so normal CLI/API imports avoid heavy LLM deps.

from typing import Any

from clearml import PipelineDecorator, Task

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import (
    CLEARML_API_ACCESS_KEY,
    CLEARML_API_HOST,
    CLEARML_API_SECRET_KEY,
    CLEARML_FILES_HOST,
    CLEARML_PROJECT_NAME,
    CLEARML_WEB_HOST,
)


def run_clearml_ollama_finetuning_pipeline(
    ollama_model: str,
    base_model_id: str | None,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
) -> dict[str, Any]:
    """Run local Ollama fine-tuning through a ClearML Pipeline.

    Args:
        ollama_model: Ollama runtime model tag.
        base_model_id: Hugging Face or local checkpoint used for QLoRA.
        dataset_path: Dataset path or Hugging Face dataset id.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        max_seq_length: Maximum sequence length.
        max_train_samples: Optional sample cap for smoke/demo runs.
        run_name: Optional ClearML run name.

    Returns:
        Fine-tuning result payload.
    """
    Task.set_credentials(
        api_host=CLEARML_API_HOST,
        web_host=CLEARML_WEB_HOST,
        files_host=CLEARML_FILES_HOST,
        key=CLEARML_API_ACCESS_KEY or None,
        secret=CLEARML_API_SECRET_KEY or None,
        store_conf_file=False,
    )
    PipelineDecorator.run_locally()
    return _ollama_finetuning_pipeline(
        ollama_model=ollama_model,
        base_model_id=base_model_id,
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        run_name=run_name,
    )


def run_clearml_managed_finetuning_pipeline(
    provider_name: str,
    provider_model: str | None,
    base_model_id: str | None,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
) -> dict[str, Any]:
    """Run managed-provider fine-tuning through a ClearML Pipeline.

    Args:
        provider_name: Provider key.
        provider_model: Provider runtime/deployment/source model override.
        base_model_id: Provider base model override.
        dataset_path: Provider dataset reference.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size, if provider uses it.
        max_seq_length: Maximum sequence length, if provider uses it.
        max_train_samples: Optional sample cap, if provider uses it.
        run_name: Optional ClearML run name.

    Returns:
        Fine-tuning result payload.
    """
    Task.set_credentials(
        api_host=CLEARML_API_HOST,
        web_host=CLEARML_WEB_HOST,
        files_host=CLEARML_FILES_HOST,
        key=CLEARML_API_ACCESS_KEY or None,
        secret=CLEARML_API_SECRET_KEY or None,
        store_conf_file=False,
    )
    PipelineDecorator.run_locally()
    return _managed_provider_finetuning_pipeline(
        provider_name=provider_name,
        provider_model=provider_model,
        base_model_id=base_model_id,
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        run_name=run_name,
    )


@PipelineDecorator.pipeline(
    name="llm-ollama-finetuning",
    project=CLEARML_PROJECT_NAME,
    version="1.0",
    add_pipeline_tags=True,
    target_project=True,
    start_controller_locally=True,
    pipeline_execution_queue=None,
    output_uri=CLEARML_FILES_HOST,
)
def _ollama_finetuning_pipeline(
    ollama_model: str,
    base_model_id: str | None,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
) -> dict[str, Any]:
    """Build the ClearML Pipeline graph for local Ollama fine-tuning."""
    result = _train_ollama_lora(
        ollama_model=ollama_model,
        base_model_id=base_model_id,
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        run_name=run_name,
    )
    _report_finetuning_controller_result("ollama", result)
    return result


@PipelineDecorator.pipeline(
    name="llm-managed-provider-finetuning",
    project=CLEARML_PROJECT_NAME,
    version="1.0",
    add_pipeline_tags=True,
    target_project=True,
    start_controller_locally=True,
    pipeline_execution_queue=None,
    output_uri=CLEARML_FILES_HOST,
)
def _managed_provider_finetuning_pipeline(
    provider_name: str,
    provider_model: str | None,
    base_model_id: str | None,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
) -> dict[str, Any]:
    """Build the ClearML Pipeline graph for managed-provider fine-tuning."""
    result = _submit_managed_provider_finetune(
        provider_name=provider_name,
        provider_model=provider_model,
        base_model_id=base_model_id,
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        run_name=run_name,
    )
    _report_finetuning_controller_result(provider_name, result)
    return result


@PipelineDecorator.component(
    name="train_ollama_lora",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _train_ollama_lora(
    ollama_model: str,
    base_model_id: str | None,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
) -> dict[str, Any]:
    """Train the local Ollama LoRA adapter inside a ClearML Pipeline step."""
    import json
    import os
    from pathlib import Path

    from src.providers.ollama import OllamaProvider

    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    provider = OllamaProvider(model=ollama_model)
    adapter_path = provider.finetune(
        dataset_path=dataset_path,
        epochs=epochs,
        run_name=run_name,
        base_model_id=base_model_id,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        clearml_enabled=True,
    )
    manifest_path = Path(adapter_path).parent / "fine_tune_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    train_metrics_path = Path(adapter_path).parent / "train_metrics.json"
    eval_metrics_path = Path(adapter_path).parent / "eval_metrics.json"
    train_metrics = (
        json.loads(train_metrics_path.read_text(encoding="utf-8"))
        if train_metrics_path.exists()
        else {}
    )
    eval_metrics = (
        json.loads(eval_metrics_path.read_text(encoding="utf-8"))
        if eval_metrics_path.exists()
        else {}
    )
    return {
        "adapter_or_job": adapter_path,
        "base_model_id": base_model_id,
        "clearml_model_id": manifest.get("clearml_model_id"),
        "clearml_dataset_id": manifest.get("clearml_dataset_id"),
        "eval_metrics": eval_metrics,
        "finetuned_ollama_model": manifest.get("finetuned_ollama_model"),
        "model": ollama_model,
        "provider": "ollama",
        "train_metrics": train_metrics,
    }


@PipelineDecorator.component(
    name="submit_managed_provider_finetune",
    return_values=["result"],
    task_type="training",
    cache=False,
    output_uri=CLEARML_FILES_HOST,
)
def _submit_managed_provider_finetune(
    provider_name: str,
    provider_model: str | None,
    base_model_id: str | None,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
) -> dict[str, Any]:
    """Submit a managed-provider fine-tuning job inside a ClearML step."""
    import json
    import os
    from pathlib import Path

    from src.providers.factory import build_llm_provider

    os.environ["CLEARML_PIPELINE_INTERNAL"] = "true"
    provider = build_llm_provider(provider_name, allow_fallback=False)
    if provider_model and hasattr(provider, "model"):
        provider.model = provider_model
    elif provider_model and hasattr(provider, "model_id"):
        provider.model_id = provider_model
    elif provider_model and hasattr(provider, "deployment"):
        provider.deployment = provider_model

    job_id = provider.finetune(
        dataset_path=dataset_path,
        epochs=epochs,
        run_name=run_name,
        base_model_id=base_model_id,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        clearml_enabled=True,
    )
    manifest_path = Path("data/runs/llm_finetuning/latest_managed_finetune.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    return {
        "adapter_or_job": job_id,
        "base_model_id": base_model_id,
        "clearml_model_id": manifest.get("clearml_model_id"),
        "clearml_dataset_id": manifest.get("clearml_dataset_id"),
        "model": provider_model or manifest.get("model_id"),
        "provider": provider_name,
    }


def _report_finetuning_controller_result(
    provider_name: str,
    result: dict[str, Any],
) -> None:
    """Report fine-tuning pipeline progress to the controller task.

    Args:
        provider_name: Fine-tuning provider key.
        result: Pipeline step result payload.
    """
    tracker = ClearMLTracker(enabled=True)
    tracker.start(
        task_name="llm-finetuning-controller",
        params={
            "controller_metric_source": "llm_finetuning",
            "provider": provider_name,
        },
        task_type="controller",
        tags=["llm", "finetuning", "controller", provider_name],
    )
    tracker.report_scalar_points(
        "pipeline_llm_finetuning/progress",
        "completed_steps",
        [(1, 1.0)],
    )
    if result.get("adapter_or_job"):
        tracker.report_scalar_points(
            "pipeline_llm_finetuning/progress",
            "submitted_or_trained_models",
            [(1, 1.0)],
        )
    for namespace in ("train_metrics", "eval_metrics"):
        metrics = result.get(namespace) or {}
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            scalar_value = _as_float(value)
            if scalar_value is None:
                continue
            tracker.report_scalar_points(
                f"pipeline_llm_finetuning/{namespace}/{metric_name}",
                provider_name,
                [(1, scalar_value)],
            )


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
