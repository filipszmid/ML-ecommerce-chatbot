"""LLM fine-tuning Click commands."""

# Click callbacks mirror CLI options. Heavy local QLoRA imports remain delayed
# so normal chat/API commands do not import training-only dependencies.

from __future__ import annotations

from typing import Any

import click

from interface.cli.utils import HELP_CONTEXT, echo_json
from master_config import CLEARML_ENABLED, OLLAMA_MODEL
from src.finetuning.clearml_pipeline import run_clearml_managed_finetuning_pipeline
from src.providers.factory import build_llm_provider

_PROVIDER_CHOICE = click.Choice(["ollama", "azure_openai", "bedrock", "vertex"])


def run_finetuning(
    provider: str = "ollama",
    model: str | None = None,
    base_model_id: str | None = None,
    data_path: str = "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
    epochs: int = 3,
    batch_size: int = 4,
    max_seq_length: int = 512,
    max_train_samples: int | None = None,
    run_name: str | None = None,
    clearml_enabled: bool = CLEARML_ENABLED,
) -> dict[str, Any]:
    """Fine-tune an LLM locally or through a provider job.

    Args:
        provider: LLM provider backend.
        model: Provider runtime model.
        base_model_id: Base model or checkpoint for local QLoRA training.
        data_path: Dataset path or Hugging Face dataset ID.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        max_seq_length: Maximum sequence length.
        max_train_samples: Optional training sample cap.
        run_name: Optional run name.
        clearml_enabled: Whether to use ClearML tracking/pipelines.

    Returns:
        Fine-tuning summary.
    """
    if clearml_enabled and provider == "ollama":
        from src.finetuning.clearml_pipeline import (
            run_clearml_ollama_finetuning_pipeline,
        )

        return run_clearml_ollama_finetuning_pipeline(
            ollama_model=model or OLLAMA_MODEL,
            base_model_id=base_model_id,
            dataset_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            run_name=run_name,
        )
    if clearml_enabled:
        return run_clearml_managed_finetuning_pipeline(
            provider_name=provider,
            provider_model=model,
            base_model_id=base_model_id,
            dataset_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            run_name=run_name,
        )

    llm_provider = build_llm_provider(provider, allow_fallback=False)
    if model and hasattr(llm_provider, "model"):
        llm_provider.model = model
    elif model and hasattr(llm_provider, "model_id"):
        llm_provider.model_id = model
    elif model and hasattr(llm_provider, "deployment"):
        llm_provider.deployment = model

    adapter_path = llm_provider.finetune(
        dataset_path=data_path,
        epochs=epochs,
        run_name=run_name,
        base_model_id=base_model_id,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        clearml_enabled=clearml_enabled,
    )

    return {
        "provider": provider,
        "model": model
        or getattr(llm_provider, "model", None)
        or getattr(llm_provider, "model_id", None)
        or getattr(llm_provider, "deployment", None),
        "base_model_id": base_model_id,
        "adapter_or_job": adapter_path,
    }


def run_adapter_registration(
    adapter_path: str,
    model: str = OLLAMA_MODEL,
    base_model_id: str | None = None,
    run_name: str | None = None,
    clearml_enabled: bool = True,
) -> dict[str, Any]:
    """Register an existing LoRA adapter without retraining.

    Args:
        adapter_path: Adapter directory path.
        model: Ollama model name.
        base_model_id: Optional base model ID.
        run_name: Optional run name.
        clearml_enabled: Whether to register in ClearML.

    Returns:
        Registration manifest.
    """
    from src.finetuning.local_qlora import register_existing_lora_adapter

    return register_existing_lora_adapter(
        adapter_path=adapter_path,
        ollama_model=model,
        base_model_id=base_model_id,
        run_name=run_name,
        clearml_enabled=clearml_enabled,
    )


@click.command(
    name="finetune-llm",
    context_settings=HELP_CONTEXT,
    help="Fine-tune a generative AI model locally or through provider jobs.",
)
@click.option("--provider", type=_PROVIDER_CHOICE, default="ollama", show_default=True)
@click.option(
    "--model",
    default=None,
    help="Provider runtime model. If omitted, provider default is used.",
)
@click.option(
    "--base-model-id",
    default=None,
    help="Hugging Face or local checkpoint for local QLoRA.",
)
@click.option(
    "--data-path",
    default="bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
    show_default=True,
    help="Path or HuggingFace ID of the dataset.",
)
@click.option(
    "--epochs",
    type=int,
    default=3,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    show_default=True,
    help="Per-device training batch size.",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=512,
    show_default=True,
    help="Maximum sequence length.",
)
@click.option(
    "--max-train-samples",
    type=int,
    default=None,
    help="Optional cap for smoke/demo runs.",
)
@click.option("--run-name", default=None, help="Optional run name for tracking.")
@click.option("--clearml/--no-clearml", default=CLEARML_ENABLED, show_default=True)
def finetune_command(
    provider: str,
    model: str | None,
    base_model_id: str | None,
    data_path: str,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_train_samples: int | None,
    run_name: str | None,
    clearml: bool,
) -> None:
    """Run the fine-tuning command.

    Args:
        provider: LLM provider backend.
        model: Provider runtime model.
        base_model_id: Base model or checkpoint.
        data_path: Dataset path or dataset ID.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        max_seq_length: Maximum sequence length.
        max_train_samples: Optional training sample cap.
        run_name: Optional run name.
        clearml: Whether to use ClearML.
    """
    echo_json(
        run_finetuning(
            provider=provider,
            model=model,
            base_model_id=base_model_id,
            data_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            run_name=run_name,
            clearml_enabled=clearml,
        )
    )


@click.command(
    name="register-llm-adapter",
    context_settings=HELP_CONTEXT,
    help="Register an existing LoRA adapter in ClearML without retraining.",
)
@click.option("--adapter-path", required=True)
@click.option("--model", default=OLLAMA_MODEL, show_default=True)
@click.option("--base-model-id", default=None)
@click.option("--run-name", default=None)
@click.option("--clearml/--no-clearml", default=True, show_default=True)
def register_llm_adapter_command(
    adapter_path: str,
    model: str,
    base_model_id: str | None,
    run_name: str | None,
    clearml: bool,
) -> None:
    """Run the adapter-registration command.

    Args:
        adapter_path: Adapter directory path.
        model: Ollama model name.
        base_model_id: Optional base model ID.
        run_name: Optional run name.
        clearml: Whether to register in ClearML.
    """
    echo_json(
        run_adapter_registration(
            adapter_path=adapter_path,
            model=model,
            base_model_id=base_model_id,
            run_name=run_name,
            clearml_enabled=clearml,
        )
    )
