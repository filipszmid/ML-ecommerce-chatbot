"""Evaluation workflows for saved classical model artifacts."""

# Evaluation workflow objects intentionally stay small, and the selection
# evaluator reuses the artifact resolver to preserve identical path semantics.

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from evaluation.reports import save_confusion_matrix_chart, save_metrics_chart
from master_config import (
    CLASSICAL_RUNS_DIR,
    DATA_DIR,
    DEFAULT_DATASET_PATH,
    EVALS_DIR,
    FEATURE_COLUMNS,
    RUNS_DIR,
)
from src.models.common.artifacts import (
    load_json,
    save_json,
    utc_timestamp,
    write_training_report,
)
from src.models.common.data import load_dataset, split_features_target
from src.models.common.metrics import (
    confusion_matrix_to_markdown,
    evaluate_classifier,
    generate_shap_summary,
)


class ArtifactEvaluationWorkflow:
    """Evaluate one saved model artifact on train/test/all splits."""

    def __init__(self, output_root: Path | str = EVALS_DIR / "jobs") -> None:
        """Initialize workflow.

        Args:
            output_root: Directory where evaluation job outputs are written.
        """
        self.output_root = Path(output_root)

    def run(
        self,
        artifact_dir: Path | str,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        split: str = "test",
    ) -> dict[str, Any]:
        """Evaluate one artifact.

        Args:
            artifact_dir: Model artifact directory.
            dataset_path: Evaluation CSV path.
            split: One of `train`, `test`, or `all`.

        Returns:
            Evaluation payload.
        """
        if split == "all":
            return {
                "artifact_dir": str(artifact_dir),
                "dataset_path": str(dataset_path),
                "split": "all",
                "evaluations": [
                    self.run(artifact_dir, dataset_path, "train"),
                    self.run(artifact_dir, dataset_path, "test"),
                ],
            }
        if split not in {"train", "test"}:
            raise ValueError("split must be one of: train, test, all")

        artifact_path = self._resolve_artifact_dir(artifact_dir)
        model = joblib.load(artifact_path / "model.joblib")
        dataframe = self._split_dataframe(
            artifact_path=artifact_path,
            dataset_path=Path(dataset_path),
            split=split,
        )

        features, target = split_features_target(dataframe)
        metrics, report, matrix = evaluate_classifier(model, features, target)
        eval_id = f"{utc_timestamp()}_{artifact_path.name}_{split}_eval"
        output_dir = self.output_root / eval_id
        output_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = output_dir / "confusion_matrix.csv"
        report_path = output_dir / "classification_report.txt"
        metrics_path = output_dir / "metrics.json"
        matrix.to_csv(matrix_path)
        report_path.write_text(report)
        save_json(metrics_path, metrics)

        confusion_chart = save_confusion_matrix_chart(
            matrix,
            output_dir / "confusion_matrix.png",
        )
        metrics_chart = save_metrics_chart(metrics, output_dir / "metrics.png")
        shap_image_path = generate_shap_summary(
            pipeline=model,
            x_val=features,
            feature_columns=FEATURE_COLUMNS,
            output_path=output_dir / "shap_summary.png",
        )
        markdown_path = output_dir / "report.md"
        write_training_report(
            output_path=markdown_path,
            model_name=str(artifact_path.name),
            run_id=eval_id,
            metrics=metrics,
            params={},
            classification_report=report,
            confusion_matrix_markdown=confusion_matrix_to_markdown(matrix),
            shap_image_path=shap_image_path,
            artifact_dir=(
                artifact_path.relative_to(DATA_DIR)
                if artifact_path.is_relative_to(DATA_DIR)
                else artifact_path
            ),
        )

        payload = {
            "eval_id": eval_id,
            "artifact_dir": str(artifact_path),
            "dataset_path": str(dataset_path),
            "split": split,
            "row_count": len(dataframe),
            "metrics": metrics,
            "report_path": str(markdown_path),
            "charts": {
                "confusion_matrix": str(confusion_chart),
                "metrics": str(metrics_chart),
                "shap_summary": str(shap_image_path) if shap_image_path else None,
            },
        }
        save_json(output_dir / "evaluation.json", payload)
        return payload

    def _split_dataframe(
        self,
        artifact_path: Path,
        dataset_path: Path,
        split: str,
    ) -> Any:
        """Load the requested split dataframe.

        Args:
            artifact_path: Model artifact directory.
            dataset_path: Dataset path.
            split: `train` or `test`.

        Returns:
            Split dataframe.
        """
        dataframe = load_dataset(dataset_path)
        indices = self._split_indices(artifact_path, split)
        if indices is None:
            return dataframe
        return dataframe.loc[indices]

    def _split_indices(self, artifact_path: Path, split: str) -> list[Any] | None:
        """Read split indices for a saved artifact.

        Args:
            artifact_path: Model artifact directory.
            split: `train` or `test`.

        Returns:
            Row indices for the split, if available.
        """
        manifest_path = artifact_path / "split_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            split_payload = manifest.get("splits", {}).get(split, {})
            indices_path = split_payload.get("indices_path")
            if indices_path:
                return load_json(artifact_path / indices_path)

        fallback_name = (
            "train_indices.json" if split == "train" else "test_indices.json"
        )
        fallback_path = artifact_path / fallback_name
        if fallback_path.exists():
            return load_json(fallback_path)
        legacy_val_path = artifact_path / "val_indices.json"
        if split == "test" and legacy_val_path.exists():
            return load_json(legacy_val_path)
        return None

    @staticmethod
    def _resolve_artifact_dir(artifact_dir: Path | str) -> Path:
        """Resolve a model artifact directory.

        Args:
            artifact_dir: Absolute path or path relative to `data/`.

        Returns:
            Resolved artifact directory.
        """
        path = Path(artifact_dir)
        if path.is_absolute():
            return path
        if path.exists():
            return path
        if path.parts and path.parts[0] == "data":
            return path
        return DATA_DIR / path

    @staticmethod
    def resolve_selection_path(selection_path: Path | str) -> Path:
        """Resolve a model-selection payload path.

        Args:
            selection_path: Absolute path, project-relative path, or data-relative path.

        Returns:
            Resolved selection payload path.
        """
        path = Path(selection_path)
        if path.is_absolute() or path.exists():
            return path
        if path.parts and path.parts[0] == "data":
            return path
        candidate = DATA_DIR / path
        if candidate.exists():
            return candidate
        legacy_path = RUNS_DIR / "latest_selection.json"
        if path.name == "latest_selection.json" and legacy_path.exists():
            return legacy_path
        return path


class SelectionEvaluationWorkflow:
    """Evaluate every model artifact listed in a model-selection payload."""

    def __init__(self, output_root: Path | str = EVALS_DIR / "jobs") -> None:
        """Initialize workflow.

        Args:
            output_root: Directory where evaluation job outputs are written.
        """
        self.output_root = Path(output_root)

    def run(
        self,
        selection_path: Path | str = CLASSICAL_RUNS_DIR / "latest_selection.json",
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        split: str = "test",
    ) -> dict[str, Any]:
        """Evaluate each run from a model-selection payload.

        Args:
            selection_path: Path to `latest_selection.json` or a selection payload.
            dataset_path: Evaluation CSV path.
            split: One of `train`, `test`, or `all`.

        Returns:
            Selection evaluation payload.
        """
        selection_file = ArtifactEvaluationWorkflow.resolve_selection_path(
            selection_path
        )
        payload = load_json(selection_file)
        selection_id = payload.get("selection_id", selection_file.stem)
        evaluator = ArtifactEvaluationWorkflow(self.output_root / selection_id)
        evaluations = [
            evaluator.run(
                artifact_dir=run["artifact_dir"],
                dataset_path=dataset_path,
                split=split,
            )
            for run in payload.get("runs", [])
        ]
        output_dir = self.output_root / selection_id
        summary = {
            "selection_id": selection_id,
            "selection_path": str(selection_file),
            "dataset_path": str(dataset_path),
            "split": split,
            "evaluations": evaluations,
        }
        save_json(output_dir / "selection_evaluation.json", summary)
        return summary
