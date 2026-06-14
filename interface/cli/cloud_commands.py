"""Cloud training Click commands."""

# Click callback signatures mirror command options for readable CLI help.

from __future__ import annotations

from typing import Any

import click

from interface.cli.utils import HELP_CONTEXT, echo_json
from master_config import (
    CLEARML_ENABLED,
    DEFAULT_CV_FOLDS,
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_EVALS,
    DEFAULT_SMOTE_K_NEIGHBORS,
    DEFAULT_USE_SMOTE,
)
from src.cloud.runner import AWSRunner, AzureRunner, CloudTrainingConfig, GCPRunner

_PROVIDER_CHOICE = click.Choice(["gcp", "aws", "azure"])


def run_cloud_training(
    provider: str = "gcp",
    models: str = "all",
    data_path: str = str(DEFAULT_DATASET_PATH),
    max_evals: int = DEFAULT_MAX_EVALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    use_smote: bool = DEFAULT_USE_SMOTE,
    smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
    clearml_enabled: bool = CLEARML_ENABLED,
    run: bool = False,
) -> dict[str, Any]:
    """Submit or render a cloud model-selection job.

    Args:
        provider: Cloud provider name.
        models: Comma-separated model names or `all`.
        data_path: Dataset CSV path.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        use_smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml_enabled: Whether to log to ClearML.
        run: Whether to submit the job instead of dry-run.

    Returns:
        Job metadata.
    """
    runners = {
        "gcp": GCPRunner,
        "aws": AWSRunner,
        "azure": AzureRunner,
    }
    config = CloudTrainingConfig(
        models=models,
        data_path=data_path,
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=clearml_enabled,
    )
    runner = runners[provider]()
    return runner.run(config, dry_run=not run)


@click.command(
    name="cloud-train",
    context_settings=HELP_CONTEXT,
    help="Run model selection remotely on cloud instances.",
)
@click.option("--provider", type=_PROVIDER_CHOICE, default="gcp", show_default=True)
@click.option("--models", default="all", show_default=True)
@click.option("--data-path", default=str(DEFAULT_DATASET_PATH), show_default=True)
@click.option("--max-evals", type=int, default=DEFAULT_MAX_EVALS, show_default=True)
@click.option("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, show_default=True)
@click.option("--smote/--no-smote", default=DEFAULT_USE_SMOTE, show_default=True)
@click.option(
    "--smote-k-neighbors",
    type=int,
    default=DEFAULT_SMOTE_K_NEIGHBORS,
    show_default=True,
)
@click.option("--clearml/--no-clearml", default=CLEARML_ENABLED, show_default=True)
@click.option(
    "--run",
    is_flag=True,
    default=False,
    help="Actually submit the job; otherwise dry run.",
)
def cloud_train_command(
    provider: str,
    models: str,
    data_path: str,
    max_evals: int,
    cv_folds: int,
    smote: bool,
    smote_k_neighbors: int,
    clearml: bool,
    run: bool,
) -> None:
    """Run the cloud training command.

    Args:
        provider: Cloud provider name.
        models: Comma-separated model names or `all`.
        data_path: Dataset CSV path.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml: Whether to log to ClearML.
        run: Whether to submit the job.
    """
    echo_json(
        run_cloud_training(
            provider=provider,
            models=models,
            data_path=data_path,
            max_evals=max_evals,
            cv_folds=cv_folds,
            use_smote=smote,
            smote_k_neighbors=smote_k_neighbors,
            clearml_enabled=clearml,
            run=run,
        )
    )
