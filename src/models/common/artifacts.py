"""Artifact utilities for model runs and reports."""

# Artifact discovery is deliberately tolerant of missing/stale selection files.
# Report writer keeps an explicit signature because it is used from multiple
# workflows with stable keyword arguments.

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from master_config import (
    CLASSICAL_RUNS_DIR,
    DATA_DIR,
    EVALS_DIR,
    FEATURE_COLUMNS,
    PRODUCT_CATEGORY_LABELS,
    PROJECT_ROOT,
    RUNS_DIR,
)


def utc_timestamp() -> str:
    """Create a compact UTC timestamp for run identifiers.

    Returns:
        Timestamp string.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_id(model_name: str) -> str:
    """Create a run identifier.

    Args:
        model_name: Model key.

    Returns:
        Run identifier.
    """
    return f"{utc_timestamp()}_{model_name}"


def ensure_artifact_dirs() -> None:
    """Create run and evaluation directories."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSICAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EVALS_DIR.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str, pipeline_id: str | None = None) -> Path:
    """Return a run artifact directory path.

    Args:
        run_id: Run identifier.
        pipeline_id: Optional parent pipeline identifier.

    Returns:
        Directory path.
    """
    if pipeline_id:
        return CLASSICAL_RUNS_DIR / pipeline_id / run_id
    return CLASSICAL_RUNS_DIR / run_id


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write JSON with stable formatting.

    Args:
        path: Output path.
        payload: JSON-serializable payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default))


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON document.

    Args:
        path: JSON path.

    Returns:
        Parsed dictionary.
    """
    return json.loads(path.read_text())


def feature_schema() -> dict[str, Any]:
    """Return the model feature schema.

    Returns:
        Feature schema payload.
    """
    return {
        "features": FEATURE_COLUMNS,
        "target": "ProductCategory",
        "target_labels": PRODUCT_CATEGORY_LABELS,
    }


def latest_model_dir(base_dir: Path = CLASSICAL_RUNS_DIR) -> Path | None:
    """Find the latest run directory with a trained model.

    Args:
        base_dir: Runs directory.

    Returns:
        Selected model directory or latest model directory, if available.
    """
    selected_dir = _selected_model_dir(base_dir)
    if selected_dir is not None:
        return selected_dir
    if not base_dir.exists():
        return None
    candidates = [
        path.parent for path in base_dir.rglob("model.joblib") if path.is_file()
    ]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    return latest.resolve()


def _selected_model_dir(base_dir: Path) -> Path | None:
    """Return the model selected by the latest model-selection run.

    Args:
        base_dir: Runs directory.

    Returns:
        Selected model directory, if it exists and has model artifacts.
    """
    selection_candidates = [
        base_dir / "latest_selection.json",
        base_dir / "classical_ml" / "latest_selection.json",
    ]
    legacy_selection_path = RUNS_DIR / "latest_selection.json"
    if legacy_selection_path not in selection_candidates:
        selection_candidates.append(legacy_selection_path)

    payload = None
    for latest_selection_path in selection_candidates:
        if not latest_selection_path.exists():
            continue
        try:
            payload = load_json(latest_selection_path)
            break
        except json.JSONDecodeError, OSError, TypeError, ValueError:
            continue
    if payload is None:
        return None
    best_artifact_dir = payload.get("best_artifact_dir")
    if not best_artifact_dir:
        return None
    candidate = Path(best_artifact_dir)
    if not candidate.is_absolute():
        parts = candidate.parts
        if parts and parts[0] == "data":
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = DATA_DIR / candidate
    if (candidate / "model.joblib").exists():
        return candidate.resolve()
    return None


def write_training_report(
    output_path: Path,
    model_name: str,
    run_id: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    classification_report: str,
    confusion_matrix_markdown: str,
    shap_image_path: Path | None = None,
    artifact_dir: Path | str | None = None,
) -> None:
    """Write a readable model evaluation report.

    Args:
        output_path: Markdown output path.
        model_name: Model key.
        run_id: Run identifier.
        metrics: Evaluation metrics.
        params: Trained parameters.
        classification_report: Text report from scikit-learn.
        confusion_matrix_markdown: Markdown confusion matrix.
        shap_image_path: Optional path to SHAP summary plot.
        artifact_dir: Optional model artifact directory to display.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_display = (
        str(artifact_dir)
        if artifact_dir is not None
        else f"data/runs/classical_ml/{run_id}"
    )
    metric_lines = "\n".join(
        f"| {name} | {value:.6f} |" for name, value in sorted(metrics.items())
    )
    param_lines = "\n".join(
        f"| {name} | `{value}` |" for name, value in sorted(params.items())
    )
    markdown_lines = [
        f"# {model_name} Evaluation",
        "",
        f"- Run: `{run_id}`",
        f"- Artifact directory: `{artifact_display}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        metric_lines,
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        param_lines,
        "",
        "## Confusion Matrix",
        "",
        confusion_matrix_markdown,
        "",
        "## Classification Report",
        "",
        "```text",
        classification_report,
        "```",
        "",
    ]

    if shap_image_path and shap_image_path.exists():
        rel_path = os.path.relpath(str(shap_image_path), str(output_path.parent))
        markdown_lines.extend(
            [
                "## Feature Interpretability (SHAP)",
                "",
                "The following plot shows the global feature importance and "
                "impact on model output:",
                "",
                f"![SHAP Summary Plot]({rel_path})",
                "",
            ]
        )

    output_path.write_text("\n".join(markdown_lines))
