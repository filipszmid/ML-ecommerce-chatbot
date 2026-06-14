"""Optional ClearML tracking wrapper."""

# ClearML is an external SDK boundary: calls can fail for transport, server,
# credential, and SDK-version reasons, and optional imports are intentionally
# delayed until tracking is used.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from clearml import OutputModel
from loguru import logger

from master_config import (
    CLEARML_API_ACCESS_KEY,
    CLEARML_API_HOST,
    CLEARML_API_SECRET_KEY,
    CLEARML_FILES_HOST,
    CLEARML_PROJECT_NAME,
    CLEARML_WEB_HOST,
)


@dataclass
class ClearMLTracker:
    """Small adapter around ClearML task logging.

    Args:
        enabled: Whether ClearML logging should be attempted.
        project_name: ClearML project name.
        api_host: ClearML API server URL.
        web_host: ClearML Web UI URL.
        files_host: ClearML file server URL.
        api_access_key: ClearML SDK access key.
        api_secret_key: ClearML SDK secret key.
    """

    enabled: bool
    project_name: str = CLEARML_PROJECT_NAME
    api_host: str = CLEARML_API_HOST
    web_host: str = CLEARML_WEB_HOST
    files_host: str = CLEARML_FILES_HOST
    api_access_key: str = CLEARML_API_ACCESS_KEY
    api_secret_key: str = CLEARML_API_SECRET_KEY

    def __post_init__(self) -> None:
        """Initialize tracker state."""
        self.task: Any | None = None
        self._logger: Any | None = None
        self._sdk_configured = False

    @property
    def task_id(self) -> str | None:
        """Return the active ClearML task id.

        Returns:
            ClearML task id, if a task was started.
        """
        if self.task is None:
            return None
        return getattr(self.task, "id", None)

    def start(
        self,
        task_name: str,
        params: dict[str, Any],
        task_type: str = "training",
        tags: list[str] | None = None,
    ) -> None:
        """Start a ClearML task when tracking is enabled.

        Args:
            task_name: ClearML task name.
            params: Connected parameters.
            task_type: ClearML task type name.
            tags: Task tags.
        """
        if not self.enabled:
            return
        try:
            task_cls = self._configure_sdk()
            current_task = task_cls.current_task()
            if current_task is not None:
                self.task = current_task
                self.task.connect(params)
                self._logger = self.task.get_logger()
                return
            clearml_task_type = getattr(
                task_cls.TaskTypes,
                task_type,
                task_cls.TaskTypes.training,
            )

            self.task = task_cls.init(
                project_name=self.project_name,
                task_name=task_name,
                task_type=clearml_task_type,
                tags=tags,
                reuse_last_task_id=False,
                output_uri=self.files_host,
                auto_connect_frameworks=False,
            )
            self.task.connect(params)
            self._logger = self.task.get_logger()
        except Exception as exc:  # pragma: no cover - depends on local ClearML server.
            logger.warning(
                "ClearML tracking disabled for this run: "
                f"{exc}. {self._credential_hint()}"
            )
            self.enabled = False

    def report_metrics(
        self, title: str, series: str, metrics: dict[str, float]
    ) -> None:
        """Report scalar metrics.

        Args:
            title: Metric title.
            series: Series name.
            metrics: Metric payload.
        """
        if not self.enabled or self._logger is None:
            return
        try:
            for metric_name, value in metrics.items():
                self._logger.report_scalar(
                    title=f"{title}/{metric_name}",
                    series=series,
                    value=value,
                    iteration=0,
                )
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("metric reporting", exc)

    def report_scalar_series(
        self,
        title: str,
        series: str,
        values: list[float],
    ) -> None:
        """Report a scalar time series.

        Args:
            title: Metric title.
            series: Series name.
            values: Ordered scalar values.
        """
        if not self.enabled or self._logger is None:
            return
        try:
            for iteration, value in enumerate(values):
                self._logger.report_scalar(
                    title=title,
                    series=series,
                    value=float(value),
                    iteration=iteration,
                )
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("scalar series reporting", exc)

    def report_scalar_points(
        self,
        title: str,
        series: str,
        points: list[tuple[int, float]],
    ) -> None:
        """Report scalar values with explicit iterations.

        Args:
            title: Metric title.
            series: Series name.
            points: Ordered `(iteration, value)` scalar points.
        """
        if not self.enabled or self._logger is None:
            return
        try:
            for iteration, value in points:
                self._logger.report_scalar(
                    title=title,
                    series=series,
                    value=float(value),
                    iteration=int(iteration),
                )
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("scalar point reporting", exc)

    def report_table(
        self,
        title: str,
        series: str,
        table: pd.DataFrame,
        iteration: int = 0,
    ) -> None:
        """Report a dataframe table.

        Args:
            title: Table title.
            series: Table series.
            table: Dataframe to report.
            iteration: Reporting iteration.
        """
        if not self.enabled or self._logger is None:
            return
        try:
            self._logger.report_table(
                title=title,
                series=series,
                iteration=iteration,
                table_plot=table,
            )
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("table reporting", exc)

    def report_confusion_matrix(
        self,
        title: str,
        series: str,
        matrix: pd.DataFrame,
        iteration: int = 0,
    ) -> None:
        """Report a confusion matrix plot.

        Args:
            title: Plot title.
            series: Plot series.
            matrix: Confusion matrix dataframe.
            iteration: Reporting iteration.
        """
        if not self.enabled or self._logger is None:
            return
        try:
            self._logger.report_confusion_matrix(
                title=title,
                series=series,
                matrix=matrix.to_numpy(),
                iteration=iteration,
                xlabels=[
                    str(column).removeprefix("pred_") for column in matrix.columns
                ],
                ylabels=[str(index).removeprefix("actual_") for index in matrix.index],
            )
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("confusion matrix reporting", exc)

    def report_image(
        self,
        title: str,
        series: str,
        image_path: Path,
        iteration: int = 0,
    ) -> None:
        """Report an image artifact.

        Args:
            title: Image title.
            series: Image series.
            image_path: Local image path.
            iteration: Reporting iteration.
        """
        if not self.enabled or self._logger is None or not image_path.exists():
            return
        try:
            self._logger.report_image(
                title=title,
                series=series,
                iteration=iteration,
                local_path=str(image_path),
            )
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("image reporting", exc)

    def report_text(self, message: str) -> None:
        """Report text to the task log.

        Args:
            message: Text message.
        """
        if not self.enabled or self._logger is None:
            return
        try:
            self._logger.report_text(message)
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable("text reporting", exc)

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        """Upload an artifact to ClearML.

        Args:
            name: Artifact name.
            artifact_object: Artifact object or path.
        """
        if not self.enabled or self.task is None:
            return
        try:
            self.task.upload_artifact(name=name, artifact_object=artifact_object)
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            self._warn_and_disable(f"artifact upload '{name}'", exc)

    def upload_artifacts(self, artifacts: dict[str, Path]) -> None:
        """Upload local file artifacts.

        Args:
            artifacts: Mapping of artifact names to local paths.
        """
        if not self.enabled:
            return
        for name, path in artifacts.items():
            if path.exists():
                self.upload_artifact(name, path)

    def register_output_model(
        self,
        model_path: Path,
        model_name: str,
        label_enumeration: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        framework: str = "scikit-learn",
    ) -> str | None:
        """Register a trained model in ClearML model registry.

        Args:
            model_path: Local serialized model path.
            model_name: ClearML model name.
            label_enumeration: Optional label mapping.
            metadata: Optional model metadata.
            framework: ClearML framework name.

        Returns:
            ClearML model id if registration succeeds.
        """
        if not self.enabled or self.task is None or not model_path.exists():
            return None
        try:

            output_model = OutputModel(
                task=self.task,
                name=model_name,
                framework=framework,
                label_enumeration=label_enumeration,
            )
            for key, value in (metadata or {}).items():
                output_model.set_metadata(key, str(value))
            output_model.update_weights(
                weights_filename=str(model_path),
                auto_delete_file=False,
                async_enable=False,
            )
            output_model.publish()
            return getattr(output_model, "id", None)
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            logger.warning(f"ClearML model registration failed: {exc}")
            return None

    def log_dataset(
        self,
        dataset_path: Path,
        dataset_name: str,
        dataset_version: str,
        tags: list[str] | None = None,
    ) -> str | None:
        """Create and upload a ClearML Dataset entry.

        Args:
            dataset_path: Local dataset file or directory path.
            dataset_name: ClearML dataset name.
            dataset_version: Dataset version.
            tags: Optional dataset tags.

        Returns:
            ClearML dataset id if creation succeeds.
        """
        if not self.enabled or not dataset_path.exists():
            return None
        try:
            self._configure_sdk()
            from clearml import Dataset

            dataset = Dataset.create(
                dataset_name=dataset_name,
                dataset_project=self.project_name,
                dataset_tags=tags,
                dataset_version=dataset_version,
                output_uri=self.files_host,
                description="Training dataset used by the ecommerce ML pipeline.",
            )
            dataset.add_files(
                path=str(dataset_path),
                recursive=dataset_path.is_dir(),
                verbose=False,
            )
            dataset.upload(
                show_progress=False,
                verbose=False,
                output_url=self.files_host,
            )
            dataset.finalize(verbose=False)
            dataset_id = getattr(dataset, "id", None)
            if self.task is not None and dataset_id:
                self.task.connect(
                    {
                        "clearml_dataset_id": dataset_id,
                        "dataset_path": str(dataset_path),
                    },
                    name="dataset",
                )
            return dataset_id
        except Exception as exc:  # pragma: no cover - depends on ClearML server.
            logger.warning(f"ClearML dataset registration failed: {exc}")
            return None

    def close(self) -> None:
        """Close the task."""
        if self.enabled and self.task is not None:
            # Inside a pipeline component the controller manages the task lifecycle.
            # Closing here would mark the task as completed before the pipeline
            # controller can store the return values, causing None results.
            import os

            if os.environ.get("CLEARML_PIPELINE_INTERNAL") == "true":
                return
            try:
                self.task.close()
            except Exception as exc:  # pragma: no cover - depends on ClearML server.
                logger.warning(f"ClearML task close failed: {exc}")

    def _configure_sdk(self) -> Any:
        """Configure ClearML SDK endpoints.

        Returns:
            ClearML Task class.
        """
        from clearml import Task

        if not self._sdk_configured:
            Task.set_credentials(
                api_host=self.api_host,
                web_host=self.web_host,
                files_host=self.files_host,
                key=self.api_access_key or None,
                secret=self.api_secret_key or None,
                store_conf_file=False,
            )
            self._sdk_configured = True
        return Task

    def _warn_and_disable(self, action: str, exc: Exception) -> None:
        """Warn about a ClearML failure and disable further tracking.

        Args:
            action: Failed action name.
            exc: Original exception.
        """
        logger.warning(f"ClearML {action} failed; tracking disabled: {exc}")
        self.enabled = False

    def _credential_hint(self) -> str:
        """Build a redacted credential status hint.

        Returns:
            Credential hint without secret values.
        """
        if self.api_access_key and self.api_secret_key:
            return (
                "SDK credentials are loaded, but ClearML rejected them. "
                "Use API credentials generated by the ClearML UI, not dashboard "
                "login text."
            )
        return (
            "SDK credentials are not loaded. Set CLEARML_API_ACCESS_KEY and "
            "CLEARML_API_SECRET_KEY in .env or export them in the shell."
        )
