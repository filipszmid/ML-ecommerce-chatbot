"""Evaluation metrics for classification runs."""

# SHAP/matplotlib imports stay lazy because explanation plots are optional and
# heavy. The broad catch keeps training/evaluation working when SHAP cannot
# explain a particular estimator.

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from master_config import PRODUCT_CATEGORY_LABELS


def evaluate_classifier(
    estimator: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, float], str, pd.DataFrame]:
    """Evaluate a fitted classifier.

    Args:
        estimator: Fitted model or pipeline.
        x_val: Validation features.
        y_val: Validation target.

    Returns:
        Metrics, classification report text, and confusion matrix dataframe.
    """
    y_pred = estimator.predict(x_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_val, y_pred, average="macro", zero_division=0)
        ),
    }
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_val)
        labels = sorted(PRODUCT_CATEGORY_LABELS)
        metrics["log_loss"] = float(log_loss(y_val, probabilities, labels=labels))

    labels = sorted(PRODUCT_CATEGORY_LABELS)
    target_names = [PRODUCT_CATEGORY_LABELS[label] for label in labels]
    report = classification_report(
        y_val,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_val, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{PRODUCT_CATEGORY_LABELS[label]}" for label in labels],
        columns=[f"pred_{PRODUCT_CATEGORY_LABELS[label]}" for label in labels],
    )
    return metrics, report, matrix_df


def confusion_matrix_to_markdown(matrix: pd.DataFrame) -> str:
    """Convert a confusion matrix dataframe into Markdown.

    Args:
        matrix: Confusion matrix dataframe.

    Returns:
        Markdown table.
    """
    headers = [""] + [str(column) for column in matrix.columns]
    separator = ["---"] * len(headers)
    rows = [
        [str(index), *[str(value) for value in matrix.loc[index].tolist()]]
        for index in matrix.index
    ]
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def probabilities_to_labels(
    probabilities: np.ndarray,
    class_labels: list[int] | np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """Map probability vectors to readable labels.

    Args:
        probabilities: Model probability matrix.
        class_labels: Optional class labels matching probability columns.

    Returns:
        Probability records sorted by model class order.
    """
    if class_labels is None:
        class_labels = list(range(probabilities.shape[1]))
    records: list[dict[str, float | str]] = []
    for class_label, probability in zip(class_labels, probabilities[0], strict=True):
        class_id = int(class_label)
        records.append(
            {
                "class_id": class_id,
                "label": PRODUCT_CATEGORY_LABELS.get(class_id, str(class_id)),
                "probability": float(probability),
            }
        )
    return sorted(
        records, key=lambda record: float(record["probability"]), reverse=True
    )


def generate_shap_summary(
    pipeline: Any,
    x_val: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> Path | None:
    """Generate and save a SHAP summary plot.

    Args:
        pipeline: Trained pipeline.
        x_val: Validation features dataframe.
        feature_columns: Feature names.
        output_path: Path to save the PNG image.

    Returns:
        Path to the saved image or None if failed.
    """
    logger = logging.getLogger(__name__)

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        shap = importlib.import_module("shap")

        logger.info("Generating SHAP summary...")

        sample_size = min(200, x_val.shape[0])
        x_val_sample = x_val.head(sample_size)

        if "scaler" in pipeline.named_steps:
            x_transformed = pipeline.named_steps["scaler"].transform(x_val_sample)
        else:
            x_transformed = x_val_sample

        classifier = pipeline.named_steps["classifier"]

        # Use TreeExplainer for tree-based models to avoid the generic
        # Explainer's Independent masker which spawns subprocesses via loky.
        # That path segfaults when combined with CatBoost/XGBoost C++ internals.
        if _is_tree_model(classifier):
            explainer = shap.TreeExplainer(classifier)
        else:
            masker = shap.maskers.Independent(x_transformed, max_samples=sample_size)
            explainer = shap.Explainer(classifier, masker)
        shap_values = explainer(x_transformed)

        plt.figure(figsize=(10, 6))
        summary_plot = shap.summary_plot
        summary_plot(
            shap_values,
            x_transformed,
            feature_names=feature_columns,
            show=False,
        )
        _save_current_matplotlib_figure(plt, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not generate SHAP values: %s", exc)
        return None


def _is_tree_model(estimator: Any) -> bool:
    """Check whether the estimator is a tree-based model supported by TreeExplainer.

    Args:
        estimator: Fitted classifier.

    Returns:
        True if TreeExplainer can handle this estimator.
    """
    tree_class_names = {
        "XGBClassifier",
        "XGBRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
    }
    return type(estimator).__name__ in tree_class_names


def _save_current_matplotlib_figure(plt_module: Any, output_path: Path) -> None:
    """Save and close the active matplotlib figure.

    Args:
        plt_module: Imported matplotlib.pyplot module.
        output_path: Plot output path.
    """
    plt_module.tight_layout()
    plt_module.savefig(output_path)
    plt_module.close()
