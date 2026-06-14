"""Azure OpenAI-compatible provider adapter."""

# Provider constructor and fine-tuning signatures mirror the shared provider
# interface and Azure job metadata.

from __future__ import annotations

import json
import os
import urllib.request

from master_config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_FINE_TUNE_MODEL,
)
from src.providers.base import LLMMessage, LLMProvider
from src.finetuning.managed_tracking import (
    finish_managed_finetune_tracking,
    log_managed_finetune_error,
    start_managed_finetune_tracking,
)


class AzureOpenAIProvider(LLMProvider):
    """LLM provider backed by Azure OpenAI-compatible chat completions."""

    provider_name = "azure_openai"

    def __init__(
        self,
        endpoint: str = AZURE_OPENAI_ENDPOINT,
        deployment: str = AZURE_OPENAI_DEPLOYMENT,
        api_version: str = AZURE_OPENAI_API_VERSION,
        fine_tune_model: str = AZURE_OPENAI_FINE_TUNE_MODEL,
        timeout_seconds: int = 30,
    ) -> None:
        """Initialize provider.

        Args:
            endpoint: Azure OpenAI endpoint.
            deployment: Deployment name.
            api_version: Azure OpenAI API version.
            fine_tune_model: Base model id used for fine-tuning jobs.
            timeout_seconds: Request timeout.
        """
        if not endpoint or not deployment:
            raise ValueError("Azure OpenAI endpoint and deployment must be configured")
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self.fine_tune_model = fine_tune_model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via Azure OpenAI.

        Args:
            messages: Chat messages.
            system_prompt: Optional system prompt.

        Returns:
            Response text.
        """
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI")
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(
            {"role": message.role, "content": message.content} for message in messages
        )
        payload = {"messages": request_messages, "temperature": 0.2, "max_tokens": 512}
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "api-key": api_key},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return raw["choices"][0]["message"]["content"]

    def finetune(
        self,
        dataset_path: str,
        epochs: int = 3,
        run_name: str | None = None,
        base_model_id: str | None = None,
        batch_size: int = 4,
        max_seq_length: int = 512,
        max_train_samples: int | None = None,
        clearml_enabled: bool = False,
    ) -> str:
        """Execute fine-tuning via Azure OpenAI REST API.

        Args:
            dataset_path: Path to training data (must be a pre-uploaded Azure file ID).
            epochs: Number of training epochs.
            run_name: Optional custom run name for tracking.
            base_model_id: Optional Azure OpenAI fine-tunable base model override.
            batch_size: Unused for Azure OpenAI fine-tuning.
            max_seq_length: Unused for Azure OpenAI fine-tuning.
            max_train_samples: Unused for Azure OpenAI fine-tuning.
            clearml_enabled: Whether to log the submitted job into ClearML.

        Returns:
            Job ID.
        """
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is required for Azure OpenAI fine-tuning"
            )

        if not dataset_path.startswith("file-"):
            raise ValueError(
                "Azure OpenAI fine-tuning requires the dataset_path to be "
                "an uploaded file ID (e.g., file-xyz)."
            )

        model_id = base_model_id or self.fine_tune_model or self.deployment
        url = f"{self.endpoint}/openai/fine_tuning/jobs?api-version={self.api_version}"
        payload = {
            "training_file": dataset_path,
            "model": model_id,
            "hyperparameters": {"n_epochs": epochs},
        }
        if run_name:
            payload["suffix"] = run_name

        tracker, run_dir, dataset_id = start_managed_finetune_tracking(
            provider_name=self.provider_name,
            dataset_path=dataset_path,
            run_name=run_name,
            params={
                "model_id": model_id,
                "deployment": self.deployment,
                "api_version": self.api_version,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_seq_length": max_seq_length,
                "max_train_samples": max_train_samples,
            },
            clearml_enabled=clearml_enabled,
        )

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            log_managed_finetune_error(
                tracker=tracker,
                run_dir=run_dir,
                provider_name=self.provider_name,
                dataset_path=dataset_path,
                model_id=model_id,
                request_payload=payload,
                error=exc,
            )
            raise

        job_id = raw.get("id", "")
        finish_managed_finetune_tracking(
            tracker=tracker,
            run_dir=run_dir,
            provider_name=self.provider_name,
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            model_id=model_id,
            job_id=job_id,
            request_payload=payload,
            response_payload=raw,
        )
        return job_id
