"""Python job wrapper for evaluating model-selection artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evaluation.workflows import SelectionEvaluationWorkflow
from master_config import CLASSICAL_RUNS_DIR, DEFAULT_DATASET_PATH


def run_selection_evaluation_job(
    selection_path: Path | str = CLASSICAL_RUNS_DIR / "latest_selection.json",
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    split: str = "test",
) -> dict[str, Any]:
    """Run selection evaluation as a Python job.

    Args:
        selection_path: Model-selection payload path.
        dataset_path: Evaluation dataset path.
        split: Dataset split to evaluate.

    Returns:
        Selection evaluation payload.
    """
    return SelectionEvaluationWorkflow().run(
        selection_path=Path(selection_path),
        dataset_path=Path(dataset_path),
        split=split,
    )


def main() -> None:
    """Run the job from environment/default values."""
    result = run_selection_evaluation_job(
        selection_path=os.getenv(
            "SELECTION_PATH",
            str(CLASSICAL_RUNS_DIR / "latest_selection.json"),
        ),
        dataset_path=os.getenv("DATA_PATH", str(DEFAULT_DATASET_PATH)),
        split=os.getenv("SPLIT", "test"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
