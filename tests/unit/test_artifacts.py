"""Tests for artifact discovery helpers."""

from __future__ import annotations

from src.models.common.artifacts import latest_model_dir, save_json


def test_latest_model_dir_prefers_selected_artifact(tmp_path) -> None:
    """Latest model discovery should use model-selection winner first."""
    runs_dir = tmp_path / "runs"
    fallback_dir = runs_dir / "fallback"
    selected_dir = runs_dir / "selection" / "best"
    fallback_dir.mkdir(parents=True)
    selected_dir.mkdir(parents=True)
    (fallback_dir / "model.joblib").write_text("fallback")
    (selected_dir / "model.joblib").write_text("selected")
    save_json(
        runs_dir / "latest_selection.json",
        {"best_artifact_dir": str(selected_dir)},
    )

    assert latest_model_dir(runs_dir) == selected_dir
