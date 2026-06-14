"""Dataset loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from master_config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load and validate the customer purchase dataset.

    Args:
        path: CSV path.

    Returns:
        Validated dataframe.

    Raises:
        FileNotFoundError: If the dataset path is missing.
        ValueError: If required columns are missing.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataframe = pd.read_csv(dataset_path)
    validate_dataset(dataframe)
    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Validate that the dataset contains the required columns.

    Args:
        dataframe: Dataset to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target.

    Args:
        dataframe: Validated dataset.

    Returns:
        Feature dataframe and target series.
    """
    return (
        dataframe[FEATURE_COLUMNS].copy(),
        dataframe[TARGET_COLUMN].astype(int).copy(),
    )


def train_validation_split(
    dataframe: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train-validation split.

    Args:
        dataframe: Validated dataset.
        test_size: Validation split ratio.
        random_state: Random seed.

    Returns:
        X_train, X_val, y_train, y_val.
    """
    features, target = split_features_target(dataframe)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
