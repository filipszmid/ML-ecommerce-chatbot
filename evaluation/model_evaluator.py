"""Backward-compatible evaluation function wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.workflows import ArtifactEvaluationWorkflow
from master_config import DEFAULT_DATASET_PATH


def evaluate_artifact(
    artifact_dir: Path | str,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a trained artifact on a dataset split.

    Args:
        artifact_dir: Model artifact directory.
        dataset_path: Evaluation CSV path.
        split: One of `train`, `test`, or `all`.

    Returns:
        Evaluation payload.
    """
    return ArtifactEvaluationWorkflow().run(
        artifact_dir=artifact_dir,
        dataset_path=dataset_path,
        split=split,
    )
