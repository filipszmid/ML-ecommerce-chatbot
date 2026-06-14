"""Main Click entrypoint for project CLI commands."""

from __future__ import annotations

import click

from interface.cli.clearml_commands import clean_clearml_command
from interface.cli.cloud_commands import cloud_train_command
from interface.cli.evaluation_commands import (
    evaluate_command,
    evaluate_selection_command,
)
from interface.cli.llm_commands import finetune_command, register_llm_adapter_command
from interface.cli.prediction_commands import predict_command
from interface.cli.training_commands import select_model_command, train_command
from interface.cli.utils import HELP_CONTEXT, run_command


@click.group(
    name="ml-ecommerce-chatbot",
    context_settings=HELP_CONTEXT,
    help="Train, evaluate, and serve ecommerce product-category models.",
)
def main() -> None:
    """Run the main CLI group."""


main.add_command(train_command)
main.add_command(select_model_command)
main.add_command(evaluate_command)
main.add_command(evaluate_selection_command)
main.add_command(predict_command)
main.add_command(finetune_command)
main.add_command(register_llm_adapter_command)
main.add_command(cloud_train_command)
main.add_command(clean_clearml_command)


def train_entrypoint() -> None:
    """Entrypoint for `poetry run train-model`."""
    run_command(train_command, prog_name="train-model")


def select_model_entrypoint() -> None:
    """Entrypoint for `poetry run select-model`."""
    run_command(select_model_command, prog_name="select-model")


def evaluate_entrypoint() -> None:
    """Entrypoint for `poetry run evaluate`."""
    run_command(evaluate_command, prog_name="evaluate")


def evaluate_selection_entrypoint() -> None:
    """Entrypoint for `poetry run evaluate-selection`."""
    run_command(evaluate_selection_command, prog_name="evaluate-selection")


def finetune_entrypoint() -> None:
    """Entrypoint for `poetry run finetune`."""
    run_command(finetune_command, prog_name="finetune")


def register_llm_adapter_entrypoint() -> None:
    """Entrypoint for `poetry run register-llm-adapter`."""
    run_command(register_llm_adapter_command, prog_name="register-llm-adapter")


def predict_entrypoint() -> None:
    """Entrypoint for `poetry run predict`."""
    run_command(predict_command, prog_name="predict")


def cloud_train_entrypoint() -> None:
    """Entrypoint for `poetry run cloud-train`."""
    run_command(cloud_train_command, prog_name="cloud-train")


def clean_clearml_entrypoint() -> None:
    """Entrypoint for `poetry run clean-clearml`."""
    run_command(clean_clearml_command, prog_name="clean-clearml")


if __name__ == "__main__":
    main()
