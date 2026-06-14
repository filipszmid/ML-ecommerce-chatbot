"""AWS Bedrock provider adapter."""

# Bedrock SDK imports are delayed until the provider is instantiated or used.
# Fine-tuning keeps the shared provider interface explicit.

from __future__ import annotations

import json

from master_config import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    BEDROCK_OUTPUT_S3_URI,
    BEDROCK_ROLE_ARN,
)
from src.providers.base import LLMMessage, LLMProvider
from src.finetuning.managed_tracking import (
    finish_managed_finetune_tracking,
    log_managed_finetune_error,
    start_managed_finetune_tracking,
)


class BedrockProvider(LLMProvider):
    """LLM provider backed by AWS Bedrock."""

    provider_name = "bedrock"

    def __init__(
        self,
        model_id: str = BEDROCK_MODEL_ID,
        region_name: str = AWS_REGION,
    ) -> None:
        """Initialize provider.

        Args:
            model_id: Bedrock model identifier.
            region_name: AWS region.
        """
        import boto3

        self.model_id = model_id
        self.runtime_client = boto3.client("bedrock-runtime", region_name=region_name)
        self.control_client = boto3.client("bedrock", region_name=region_name)

    def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via Bedrock.

        Args:
            messages: Chat messages.
            system_prompt: Optional system prompt.

        Returns:
            Response text.
        """
        if self.model_id.startswith("anthropic."):
            return self._generate_anthropic(messages, system_prompt)
        return self._generate_meta(messages, system_prompt)

    def _generate_anthropic(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None,
    ) -> str:
        """Generate with an Anthropic Bedrock model.

        Args:
            messages: Chat messages.
            system_prompt: Optional system prompt.

        Returns:
            Generated text.
        """
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "system": system_prompt or "",
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
                if message.role in {"user", "assistant"}
            ],
        }
        response = self.runtime_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload),
        )
        parsed = json.loads(response["body"].read())
        return "".join(block.get("text", "") for block in parsed.get("content", []))

    def _generate_meta(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None,
    ) -> str:
        """Generate with a Meta/Llama Bedrock model.

        Args:
            messages: Chat messages.
            system_prompt: Optional system prompt.

        Returns:
            Generated text.
        """
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        prompt_parts.extend(
            f"{message.role.title()}: {message.content}" for message in messages
        )
        prompt_parts.append("Assistant:")
        payload = {
            "prompt": "\n".join(prompt_parts),
            "max_gen_len": 512,
            "temperature": 0.2,
            "top_p": 0.9,
        }
        response = self.runtime_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload),
        )
        parsed = json.loads(response["body"].read())
        return parsed.get("generation", "")

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
        """Execute fine-tuning via AWS Bedrock Model Customization Jobs.

        Args:
            dataset_path: Path to training data
                (must be an S3 URI, e.g., s3://bucket/data.jsonl).
            epochs: Number of training epochs.
            run_name: Optional custom run name for tracking.
            base_model_id: Optional Bedrock base model override.
            batch_size: Unused for Bedrock fine-tuning.
            max_seq_length: Unused for Bedrock fine-tuning.
            max_train_samples: Unused for Bedrock fine-tuning.
            clearml_enabled: Whether to log the submitted job into ClearML.

        Returns:
            Job ARN.
        """
        import time

        base_model_identifier = base_model_id or self.model_id

        if not dataset_path.startswith("s3://"):
            raise ValueError(
                "Bedrock fine-tuning requires the dataset_path to be an S3 URI."
            )

        role_arn = BEDROCK_ROLE_ARN
        output_uri = BEDROCK_OUTPUT_S3_URI

        if not role_arn or not output_uri:
            raise ValueError(
                "BEDROCK_ROLE_ARN and BEDROCK_OUTPUT_S3_URI environment "
                "variables must be set."
            )

        job_name = run_name or f"finetune-{int(time.time())}"
        normalized_model_name = base_model_identifier.replace(":", "-").replace(
            ".", "-"
        )
        custom_model_name = f"custom-{normalized_model_name}-{int(time.time())}"
        request_payload = {
            "jobName": job_name,
            "customModelName": custom_model_name,
            "roleArn": role_arn,
            "baseModelIdentifier": base_model_identifier,
            "trainingDataConfig": {"s3Uri": dataset_path},
            "outputDataConfig": {"s3Uri": output_uri},
            "hyperParameters": {
                "epochCount": str(epochs),
                "batchSize": "1",
                "learningRate": "0.00001",
            },
        }
        tracker, run_dir, dataset_id = start_managed_finetune_tracking(
            provider_name=self.provider_name,
            dataset_path=dataset_path,
            run_name=job_name,
            params={
                "model_id": base_model_identifier,
                "job_name": job_name,
                "custom_model_name": custom_model_name,
                "output_uri": output_uri,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_seq_length": max_seq_length,
                "max_train_samples": max_train_samples,
            },
            clearml_enabled=clearml_enabled,
        )
        try:
            response = self.control_client.create_model_customization_job(
                **request_payload
            )
        except Exception as exc:
            log_managed_finetune_error(
                tracker=tracker,
                run_dir=run_dir,
                provider_name=self.provider_name,
                dataset_path=dataset_path,
                model_id=base_model_identifier,
                request_payload=request_payload,
                error=exc,
            )
            raise

        job_arn = response.get("jobArn", "")
        finish_managed_finetune_tracking(
            tracker=tracker,
            run_dir=run_dir,
            provider_name=self.provider_name,
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            model_id=base_model_identifier,
            job_id=job_arn,
            request_payload=request_payload,
            response_payload=response,
        )
        return job_arn
