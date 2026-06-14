"""Cloud training job wrappers for GCP, AWS, and Azure."""

# Cloud SDKs are optional and provider-specific; imports stay inside `run`
# methods so local development does not import every cloud dependency.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from master_config import (
    CLEARML_API_HOST,
    CLEARML_FILES_HOST,
    CLEARML_SERVING_BASE_URL,
    CLEARML_SERVING_ENDPOINT,
    CLEARML_WEB_HOST,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudTrainingConfig:
    """Cloud model-selection job configuration.

    Args:
        models: Comma-separated model names or `all`.
        data_path: Dataset CSV path.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        use_smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml_enabled: Whether to log to ClearML.
    """

    models: str
    data_path: str
    max_evals: int
    cv_folds: int
    use_smote: bool
    smote_k_neighbors: int
    clearml_enabled: bool


class CloudRunner(ABC):
    """Base class for cloud training runners."""

    def __init__(
        self, image_uri: str = "gcr.io/placeholder-project/ml-ecommerce-trainer"
    ):
        """Initialize the cloud runner.

        Args:
            image_uri: Docker image URI containing the training code.
        """
        self.image_uri = image_uri

    def build_command(self, config: CloudTrainingConfig) -> list[str]:
        """Construct the CLI command for the container.

        Args:
            config: Cloud training config.

        Returns:
            Command array.
        """
        cmd = [
            "ml-ecommerce-chatbot",
            "select-model",
            "--models",
            config.models,
            "--data-path",
            config.data_path,
            "--max-evals",
            str(config.max_evals),
            "--cv-folds",
            str(config.cv_folds),
            "--smote-k-neighbors",
            str(config.smote_k_neighbors),
        ]
        if config.use_smote:
            cmd.append("--smote")
        else:
            cmd.append("--no-smote")
        if config.clearml_enabled:
            cmd.append("--clearml")
        return cmd

    def build_environment(self, config: CloudTrainingConfig) -> dict[str, str]:
        """Construct environment variables for the training container.

        Args:
            config: Cloud training config.

        Returns:
            Environment variable mapping.
        """
        if not config.clearml_enabled:
            return {}
        return {
            "CLEARML_ENABLED": "true",
            "CLEARML_API_HOST": CLEARML_API_HOST,
            "CLEARML_WEB_HOST": CLEARML_WEB_HOST,
            "CLEARML_FILES_HOST": CLEARML_FILES_HOST,
            "CLEARML_SERVING_BASE_URL": CLEARML_SERVING_BASE_URL,
            "CLEARML_SERVING_ENDPOINT": CLEARML_SERVING_ENDPOINT,
        }

    @abstractmethod
    def run(self, config: CloudTrainingConfig, dry_run: bool = True) -> dict[str, Any]:
        """Trigger the cloud training job.

        Args:
            config: Cloud training config.
            dry_run: If True, do not actually trigger the job.

        Returns:
            Job metadata.
        """
        raise NotImplementedError


class GCPRunner(CloudRunner):
    """Google Cloud Vertex AI training runner."""

    def run(self, config: CloudTrainingConfig, dry_run: bool = True) -> dict[str, Any]:
        """Trigger or render a Vertex AI training job.

        Args:
            config: Cloud training config.
            dry_run: If True, do not submit the job.

        Returns:
            Job metadata.
        """
        from google.cloud import aiplatform

        command = self.build_command(config)
        environment = self.build_environment(config)
        job_display_name = "ml-ecommerce-model-selection"

        logger.info(
            "[GCP Vertex AI] Preparing CustomContainerTrainingJob: %s",
            job_display_name,
        )
        logger.info("[GCP Vertex AI] Image: %s", self.image_uri)
        logger.info("[GCP Vertex AI] Command: %s", " ".join(command))

        if not dry_run:
            logger.info("Initializing Vertex AI SDK...")
            aiplatform.init()
            job = aiplatform.CustomContainerTrainingJob(
                display_name=job_display_name,
                container_uri=self.image_uri,
                command=command,
            )
            logger.info("Submitting job...")
            job.run(
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                environment_variables=environment,
            )
            return {"provider": "gcp", "status": "submitted", "job_name": job.name}

        return {
            "provider": "gcp",
            "status": "dry_run",
            "command": command,
            "environment": environment,
        }


class AWSRunner(CloudRunner):
    """Amazon SageMaker training runner."""

    def run(self, config: CloudTrainingConfig, dry_run: bool = True) -> dict[str, Any]:
        """Trigger or render a SageMaker training job.

        Args:
            config: Cloud training config.
            dry_run: If True, do not submit the job.

        Returns:
            Job metadata.
        """
        import sagemaker
        from sagemaker.estimator import Estimator

        command = self.build_command(config)
        environment = self.build_environment(config)
        job_name = "ml-ecommerce-model-selection"

        logger.info("[AWS SageMaker] Preparing Estimator: %s", job_name)
        logger.info("[AWS SageMaker] Image: %s", self.image_uri)
        logger.info("[AWS SageMaker] Command: %s", " ".join(command))

        if not dry_run:
            logger.info("Initializing SageMaker session...")
            sagemaker_session = sagemaker.Session()
            role = sagemaker.get_execution_role()

            # Using Estimator for custom container
            estimator = Estimator(
                image_uri=self.image_uri,
                role=role,
                instance_count=1,
                instance_type="ml.m5.2xlarge",
                sagemaker_session=sagemaker_session,
                container_entry_point=command,
                environment=environment,
            )
            logger.info("Submitting job...")
            estimator.fit(job_name=job_name)
            return {"provider": "aws", "status": "submitted", "job_name": job_name}

        return {
            "provider": "aws",
            "status": "dry_run",
            "command": command,
            "environment": environment,
        }


class AzureRunner(CloudRunner):
    """Azure Machine Learning training runner."""

    def run(self, config: CloudTrainingConfig, dry_run: bool = True) -> dict[str, Any]:
        """Trigger or render an Azure ML training job.

        Args:
            config: Cloud training config.
            dry_run: If True, do not submit the job.

        Returns:
            Job metadata.
        """
        from azure.ai.ml import MLClient, command
        from azure.identity import DefaultAzureCredential

        cmd_list = self.build_command(config)
        environment_variables = self.build_environment(config)
        cmd_str = " ".join(cmd_list)
        job_name = "ml-ecommerce-model-selection"

        logger.info("[Azure ML] Preparing Command Job: %s", job_name)
        logger.info("[Azure ML] Environment (Image): %s", self.image_uri)
        logger.info("[Azure ML] Command: %s", cmd_str)

        if not dry_run:
            logger.info("Initializing Azure ML Client...")
            credential = DefaultAzureCredential()
            ml_client = MLClient.from_config(credential=credential)

            job = command(
                code="./",
                command=cmd_str,
                environment=f"{self.image_uri}@latest",
                environment_variables=environment_variables,
                compute="cpu-cluster",
                display_name=job_name,
            )
            logger.info("Submitting job...")
            returned_job = ml_client.jobs.create_or_update(job)
            return {
                "provider": "azure",
                "status": "submitted",
                "job_name": returned_job.name,
            }

        return {
            "provider": "azure",
            "status": "dry_run",
            "command": cmd_str,
            "environment": environment_variables,
        }
