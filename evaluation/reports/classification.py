"""Classification evaluation chart writers."""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd


def save_confusion_matrix_chart(matrix: pd.DataFrame, output_path: Path) -> Path:
    """Save a confusion matrix heatmap.

    Args:
        matrix: Confusion matrix dataframe.
        output_path: PNG output path.

    Returns:
        Saved chart path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 6))
    FigureCanvasAgg(figure)
    axis: Axes = figure.add_subplot(1, 1, 1)
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    for row_index, row in enumerate(matrix.to_numpy()):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path)
    return output_path


def save_metrics_chart(metrics: dict[str, float], output_path: Path) -> Path:
    """Save a metric bar chart.

    Args:
        metrics: Metric name/value mapping.
        output_path: PNG output path.

    Returns:
        Saved chart path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(metrics)
    metric_values = [float(metrics[name]) for name in metric_names]
    figure = Figure(figsize=(10, 5))
    FigureCanvasAgg(figure)
    axis: Axes = figure.add_subplot(1, 1, 1)
    axis.bar(metric_names, metric_values, color="#2f6f9f")
    axis.set_title("Evaluation Metrics")
    axis.set_ylabel("Value")
    axis.set_ylim(0, max([1.0, *metric_values]))
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path)
    return output_path
