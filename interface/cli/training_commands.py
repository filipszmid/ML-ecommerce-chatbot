"""Training and model-selection Click commands."""

# Click callbacks mirror CLI options and public workflow parameters.

from __future__ import annotations

from pathlib import Path
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
from src.models.registry import get_workflow
from src.models.selection import ModelSelectionWorkflow


def run_training(
    model: str,
    data_path: Path | str = DEFAULT_DATASET_PATH,
    run_name: str | None = None,
    max_evals: int = DEFAULT_MAX_EVALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    use_smote: bool = DEFAULT_USE_SMOTE,
    smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
    clearml_enabled: bool = CLEARML_ENABLED,
) -> dict[str, Any]:
    """Train one classical model workflow.

    Args:
        model: Registered model name.
        data_path: Dataset CSV path.
        run_name: Optional run name.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        use_smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml_enabled: Whether to log to ClearML.

    Returns:
        Training summary.
    """
    workflow = get_workflow(model)
    result = workflow.run(
        dataset_path=Path(data_path),
        run_name=run_name,
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=clearml_enabled,
    )
    return {
        "model_name": result.model_name,
        "run_id": result.run_id,
        "artifact_dir": str(result.artifact_dir),
        "metrics": result.metrics,
        "report_path": str(result.report_path),
    }


def run_model_selection(
    models: str = "all",
    data_path: Path | str = DEFAULT_DATASET_PATH,
    max_evals: int = DEFAULT_MAX_EVALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    use_smote: bool = DEFAULT_USE_SMOTE,
    smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
    clearml_enabled: bool = CLEARML_ENABLED,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Run model selection across registered workflows.

    Args:
        models: Comma-separated model names or `all`.
        data_path: Dataset CSV path.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        use_smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml_enabled: Whether to log to ClearML.
        continue_on_error: Continue when one model fails.

    Returns:
        Selection summary.
    """
    workflow = ModelSelectionWorkflow()
    return workflow.run(
        models=models,
        dataset_path=Path(data_path),
        max_evals=max_evals,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        clearml_enabled=clearml_enabled,
        continue_on_error=continue_on_error,
    )


@click.command(name="train", context_settings=HELP_CONTEXT, help="Train one model.")
@click.option("--model", required=True)
@click.option("--data-path", default=str(DEFAULT_DATASET_PATH), show_default=True)
@click.option("--run-name", default=None)
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
def train_command(
    model: str,
    data_path: str,
    run_name: str | None,
    max_evals: int,
    cv_folds: int,
    smote: bool,
    smote_k_neighbors: int,
    clearml: bool,
) -> None:
    """Run the train command.

    Args:
        model: Registered model name.
        data_path: Dataset CSV path.
        run_name: Optional run name.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml: Whether to log to ClearML.
    """
    echo_json(
        run_training(
            model=model,
            data_path=data_path,
            run_name=run_name,
            max_evals=max_evals,
            cv_folds=cv_folds,
            use_smote=smote,
            smote_k_neighbors=smote_k_neighbors,
            clearml_enabled=clearml,
        )
    )


@click.command(
    name="select-model", context_settings=HELP_CONTEXT, help="Run model selection."
)
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
@click.option("--continue-on-error", is_flag=True, default=False)
def select_model_command(
    models: str,
    data_path: str,
    max_evals: int,
    cv_folds: int,
    smote: bool,
    smote_k_neighbors: int,
    clearml: bool,
    continue_on_error: bool,
) -> None:
    """Run the model-selection command.

    Args:
        models: Comma-separated model names or `all`.
        data_path: Dataset CSV path.
        max_evals: Hyperparameter search budget.
        cv_folds: Cross-validation folds.
        smote: Whether to apply SMOTE.
        smote_k_neighbors: SMOTE neighbor count.
        clearml: Whether to log to ClearML.
        continue_on_error: Continue when one model fails.
    """
    echo_json(
        run_model_selection(
            models=models,
            data_path=data_path,
            max_evals=max_evals,
            cv_folds=cv_folds,
            use_smote=smote,
            smote_k_neighbors=smote_k_neighbors,
            clearml_enabled=clearml,
            continue_on_error=continue_on_error,
        )
    )
