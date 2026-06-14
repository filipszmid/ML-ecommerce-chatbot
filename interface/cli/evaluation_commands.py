"""Evaluation Click commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from evaluation.model_evaluator import evaluate_artifact
from evaluation.workflows import SelectionEvaluationWorkflow
from interface.cli.utils import HELP_CONTEXT, echo_json
from master_config import CLASSICAL_RUNS_DIR, DEFAULT_DATASET_PATH

_SPLIT_CHOICE = click.Choice(["train", "test", "all"])


def run_artifact_evaluation(
    model_dir: Path | str,
    data_path: Path | str = DEFAULT_DATASET_PATH,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate one trained artifact.

    Args:
        model_dir: Model artifact directory.
        data_path: Evaluation dataset path.
        split: Dataset split to evaluate.

    Returns:
        Evaluation summary.
    """
    return evaluate_artifact(
        artifact_dir=Path(model_dir),
        dataset_path=Path(data_path),
        split=split,
    )


def run_selection_evaluation(
    selection_path: Path | str = CLASSICAL_RUNS_DIR / "latest_selection.json",
    data_path: Path | str = DEFAULT_DATASET_PATH,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate every artifact in a model-selection payload.

    Args:
        selection_path: Model-selection payload path.
        data_path: Evaluation dataset path.
        split: Dataset split to evaluate.

    Returns:
        Selection evaluation summary.
    """
    return SelectionEvaluationWorkflow().run(
        selection_path=Path(selection_path),
        dataset_path=Path(data_path),
        split=split,
    )


@click.command(
    name="evaluate", context_settings=HELP_CONTEXT, help="Evaluate an artifact."
)
@click.option("--model-dir", required=True)
@click.option("--data-path", default=str(DEFAULT_DATASET_PATH), show_default=True)
@click.option("--split", type=_SPLIT_CHOICE, default="test", show_default=True)
def evaluate_command(model_dir: str, data_path: str, split: str) -> None:
    """Run artifact evaluation from the CLI.

    Args:
        model_dir: Model artifact directory.
        data_path: Evaluation dataset path.
        split: Dataset split to evaluate.
    """
    echo_json(
        run_artifact_evaluation(
            model_dir=model_dir,
            data_path=data_path,
            split=split,
        )
    )


@click.command(
    name="evaluate-selection",
    context_settings=HELP_CONTEXT,
    help="Evaluate every run from a model-selection payload.",
)
@click.option(
    "--selection-path",
    default=str(CLASSICAL_RUNS_DIR / "latest_selection.json"),
    show_default=True,
)
@click.option("--data-path", default=str(DEFAULT_DATASET_PATH), show_default=True)
@click.option("--split", type=_SPLIT_CHOICE, default="test", show_default=True)
def evaluate_selection_command(selection_path: str, data_path: str, split: str) -> None:
    """Run selection evaluation from the CLI.

    Args:
        selection_path: Model-selection payload path.
        data_path: Evaluation dataset path.
        split: Dataset split to evaluate.
    """
    echo_json(
        run_selection_evaluation(
            selection_path=selection_path,
            data_path=data_path,
            split=split,
        )
    )
