# pyright: reportGeneralTypeIssues=false
"""
Data ingestion — loads NASA C-MAPSS data and applies AES-256 encryption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import pandas as pd

from pulsenet.core.exceptions import DataError
from pulsenet.logger import get_logger

log = get_logger(__name__)

# Standard C-MAPSS column names
CMAPSS_COLUMNS: list[str] = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Sensors known to be flat / noisy in FD001
DEFAULT_DROP_COLS: list[str] = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def load_raw(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load raw C-MAPSS whitespace-separated file with validation."""
    path = Path(filepath)
    if not path.exists():
        raise DataError(f"Data file not found: {path}")

    try:
        # names argument expects a list[str]
        df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)

        # --- Data validation ---
        n_nan = int(df.isna().sum().sum())
        # select_dtypes returns a DataFrame, so we can check it
        numeric_df = df.select_dtypes(include=[np.number])
        n_inf = int(np.isinf(numeric_df).sum().sum())

        if n_nan > 0:
            log.warning(
                "NaN values detected in raw data — filling with column median",
                extra={"nan_count": n_nan, "file": path.name},
            )
            df = df.fillna(df.median(numeric_only=True))

        if n_inf > 0:
            log.warning(
                "Infinite values detected — clipping to column min/max",
                extra={"inf_count": n_inf, "file": path.name},
            )
            for col in numeric_df.columns:
                col_data = df[col]
                finite_mask = np.isfinite(col_data)
                finite_vals = col_data[finite_mask]
                if not finite_vals.empty:  # type: ignore
                    df[col] = col_data.clip(
                        lower=finite_vals.min(), upper=finite_vals.max()
                    )

        if len(df.columns) != len(CMAPSS_COLUMNS):
            log.error(
                "Column count mismatch",
                extra={"expected": len(CMAPSS_COLUMNS), "got": len(df.columns)},
            )
            raise DataError(
                f"Expected {len(CMAPSS_COLUMNS)} columns, got {len(df.columns)}"
            )

        log.info(
            "Raw data loaded & validated",
            extra={
                "file": path.name,
                "rows": len(df),
                "nan_filled": n_nan,
                "inf_clipped": n_inf,
            },
        )
        return df
    except Exception as e:
        if isinstance(e, DataError):
            raise
        raise DataError(f"Failed to load raw data from {path}: {e}") from e


def load_rul(filepath: Union[str, Path]) -> pd.Series:
    """Load ground-truth RUL file."""
    path = Path(filepath)
    if not path.exists():
        raise DataError(f"RUL file not found: {path}")

    try:
        # Cast to Series explicitly
        rul_df = pd.read_csv(path, header=None, names=["RUL"])
        rul = cast(pd.Series, rul_df["RUL"])
        log.info("RUL loaded", extra={"units": len(rul)})
        return rul
    except Exception as e:
        raise DataError(f"Failed to load RUL from {path}: {e}") from e


def drop_noisy_columns(
    df: pd.DataFrame,
    drop_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Drop constant/noisy columns."""
    cols = drop_cols or DEFAULT_DROP_COLS

    try:
        to_drop = [c for c in cols if c in df.columns]
        df = df.drop(columns=to_drop, errors="ignore")
        log.info(
            "Dropped noisy columns",
            extra={"dropped": len(to_drop), "remaining": len(df.columns)},
        )
        return df
    except Exception as e:
        raise DataError(f"Failed to drop noisy columns: {e}") from e


def ingest(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
    drop_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full ingestion: load + clean both train and test sets."""
    train_df = drop_noisy_columns(load_raw(train_path), drop_cols)
    test_df = drop_noisy_columns(load_raw(test_path), drop_cols)
    log.info(
        "Ingestion complete",
        extra={"train_rows": len(train_df), "test_rows": len(test_df)},
    )
    return train_df, test_df
