"""Google Vertex AI provider adapter."""

# Vertex SDK imports are delayed until the provider is instantiated or used.
# Fine-tuning keeps the shared provider interface explicit.

from __future__ import annotations

from master_config import GCP_LOCATION, GCP_PROJECT_ID, VERTEX_MODEL_ID
from src.providers.base import LLMMessage, LLMProvider
from src.finetuning.managed_tracking import (
    finish_managed_finetune_tracking,
    log_managed_finetune_error,
    start_managed_finetune_tracking,
)


class VertexAIProvider(LLMProvider):
    """LLM provider backed by Vertex AI generative models."""

    provider_name = "vertex"

    def __init__(
        self,
        project_id: str = GCP_PROJECT_ID,
        location: str = GCP_LOCATION,
        model_id: str = VERTEX_MODEL_ID,
    ) -> None:
        """Initialize provider.

        Args:
            project_id: GCP project identifier.
            location: Vertex AI location.
            model_id: Vertex generative model identifier.
        """
        if not project_id:
            raise ValueError("GCP_PROJECT_ID must be configured for Vertex AI")
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project_id, location=location)
        self.model_id = model_id
        self.model = GenerativeModel(model_id)

    def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via Vertex AI.

        Args:
            messages: Chat messages.
            system_prompt: Optional system prompt.

        Returns:
            Response text.
        """
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        prompt_parts.extend(
            f"{message.role}: {message.content}" for message in messages
        )
        response = self.model.generate_content("\n".join(prompt_parts))
        return response.text or ""

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
        """Execute fine-tuning for Google Vertex AI Generative Models.

        Args:
            dataset_path: Path to training data
                (must be a GCS URI like gs://bucket/dataset.jsonl).
            epochs: Number of training epochs.
            run_name: Optional custom run name for tracking.
            base_model_id: Optional Vertex source model override.
            batch_size: Unused for Vertex AI fine-tuning.
            max_seq_length: Unused for Vertex AI fine-tuning.
            max_train_samples: Unused for Vertex AI fine-tuning.
            clearml_enabled: Whether to log the submitted job into ClearML.

        Returns:
            Job ID.
        """
        source_model = base_model_id or self.model_id

        if not dataset_path.startswith("gs://"):
            raise ValueError(
                "Vertex AI fine-tuning requires the dataset_path to be a GCS URI "
                "(e.g., gs://...)."
            )

        try:
            from vertexai.tuning import sft
        except ImportError:
            from vertexai.preview.tuning import sft

        display_name = run_name or f"finetune-{source_model}"
        request_payload = {
            "source_model": source_model,
            "train_dataset": dataset_path,
            "epochs": epochs,
            "adapter_size": 16,
            "tuned_model_display_name": display_name,
        }
        tracker, run_dir, dataset_id = start_managed_finetune_tracking(
            provider_name=self.provider_name,
            dataset_path=dataset_path,
            run_name=display_name,
            params={
                "model_id": source_model,
                "display_name": display_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_seq_length": max_seq_length,
                "max_train_samples": max_train_samples,
            },
            clearml_enabled=clearml_enabled,
        )
        try:
            sft_tuning_job = sft.train(**request_payload)
        except Exception as exc:
            log_managed_finetune_error(
                tracker=tracker,
                run_dir=run_dir,
                provider_name=self.provider_name,
                dataset_path=dataset_path,
                model_id=source_model,
                request_payload=request_payload,
                error=exc,
            )
            raise

        job_id = getattr(sft_tuning_job, "name", "")
        finish_managed_finetune_tracking(
            tracker=tracker,
            run_dir=run_dir,
            provider_name=self.provider_name,
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            model_id=source_model,
            job_id=job_id,
            request_payload=request_payload,
            response_payload={
                "name": job_id,
                "resource_name": getattr(sft_tuning_job, "resource_name", ""),
                "tuned_model_name": getattr(sft_tuning_job, "tuned_model_name", ""),
            },
        )
        return job_id
