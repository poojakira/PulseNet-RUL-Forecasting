# pyright: reportGeneralTypeIssues=false
"""
Feature engineering & normalization for sensor data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pulsenet.core.exceptions import DataError
from pulsenet.logger import get_logger

log = get_logger(__name__)


def compute_rolling_features(
    df: pd.DataFrame,
    window: int = 5,
    sensor_prefix: str = "sensor_",
) -> pd.DataFrame:
    """Add rolling mean features for every sensor column."""
    try:
        sensor_cols = [c for c in df.columns if str(c).startswith(sensor_prefix)]
        for col in sensor_cols:
            # Explicitly cast to Series for type safety
            df[f"{col}_rolling_mean"] = df.groupby("unit_number")[col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        log.info(
            "Rolling features computed",
            extra={"window": window, "sensors": len(sensor_cols)},
        )
        return df
    except Exception as e:
        raise DataError(f"Failed to compute rolling features: {e}") from e


def normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """MinMax normalize features, fit on train, transform test."""
    try:
        exclude = set(exclude_cols or ["unit_number", "time_in_cycles"])
        feat_cols = [c for c in train_df.columns if c not in exclude]

        scaler = MinMaxScaler()
        train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])
        log.info("Normalization complete", extra={"features": len(feat_cols)})
        return train_df, test_df, scaler
    except Exception as e:
        raise DataError(f"Normalization failed: {e}") from e


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excluding metadata)."""
    return [
        str(c)
        for c in df.columns
        if str(c) not in ("unit_number", "time_in_cycles", "is_anomaly")
    ]


def create_labels(
    test_df: pd.DataFrame,
    rul: pd.Series,
    failure_threshold: int = 30,
) -> np.ndarray:
    """Create binary labels from RUL: 1 = failing (RUL ≤ threshold)."""
    try:
        y_true: list[int] = []
        max_cycles = test_df.groupby("unit_number")["time_in_cycles"].max()

        for unit_id in test_df["unit_number"].unique():
            idx = int(unit_id)
            unit_data = test_df[test_df["unit_number"] == unit_id]

            # rul is a Series, index needs to be handled correctly
            final_rul = float(rul.iloc[idx - 1])
            max_c = float(max_cycles[unit_id])  # type: ignore

            for cycle in unit_data["time_in_cycles"]:
                current_rul = final_rul + (max_c - float(cycle))
                y_true.append(1 if current_rul <= failure_threshold else 0)

        return np.array(y_true)
    except Exception as e:
        raise DataError(f"Failed to create labels: {e}") from e


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 30,
) -> np.ndarray:
    """Create sliding-window sequences for LSTM / Transformer models."""
    try:
        sequences: list[np.ndarray] = []
        for uid in df["unit_number"].unique():
            unit = df[df["unit_number"] == uid][feature_cols].values  # type: ignore
            if len(unit) >= seq_len:
                for i in range(len(unit) - seq_len + 1):
                    sequences.append(unit[i : i + seq_len])  # type: ignore

        if not sequences:
            return np.empty((0, seq_len, len(feature_cols)))

        arr = np.array(sequences)
        log.info("Sequences created", extra={"count": len(arr), "seq_len": seq_len})
        return arr
    except Exception as e:
        raise DataError(f"Failed to create sequences: {e}") from e


def preprocess_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rolling_window: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Full preprocessing: rolling features + normalization."""
    train_df = compute_rolling_features(train_df.copy(), window=rolling_window)
    test_df = compute_rolling_features(test_df.copy(), window=rolling_window)
    train_df, test_df, scaler = normalize(train_df, test_df)
    return train_df, test_df, scaler
