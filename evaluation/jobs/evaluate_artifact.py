"""Python job wrapper for evaluating one model artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evaluation.workflows import ArtifactEvaluationWorkflow
from master_config import DEFAULT_DATASET_PATH


def run_artifact_evaluation_job(
    artifact_dir: Path | str,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    split: str = "test",
) -> dict[str, Any]:
    """Run artifact evaluation as a Python job.

    Args:
        artifact_dir: Model artifact directory.
        dataset_path: Evaluation dataset path.
        split: Dataset split to evaluate.

    Returns:
        Evaluation payload.
    """
    return ArtifactEvaluationWorkflow().run(
        artifact_dir=Path(artifact_dir),
        dataset_path=Path(dataset_path),
        split=split,
    )


def main() -> None:
    """Run the job from environment/default values."""
    artifact_dir = os.getenv("MODEL_DIR") or os.getenv("ARTIFACT_DIR")
    if not artifact_dir:
        raise ValueError(
            "MODEL_DIR or ARTIFACT_DIR is required for artifact evaluation jobs."
        )

    result = run_artifact_evaluation_job(
        artifact_dir=artifact_dir,
        dataset_path=os.getenv("DATA_PATH", str(DEFAULT_DATASET_PATH)),
        split=os.getenv("SPLIT", "test"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
