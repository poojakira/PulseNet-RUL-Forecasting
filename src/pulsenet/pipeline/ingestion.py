"""
Data ingestion — loads NASA C-MAPSS data and applies AES-256 encryption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pulsenet.logger import get_logger

log = get_logger(__name__)

# Standard C-MAPSS column names
CMAPSS_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Sensors known to be flat / noisy in FD001
DEFAULT_DROP_COLS = [
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


def load_raw(filepath: str | Path) -> pd.DataFrame:
    """Load raw C-MAPSS whitespace-separated file with validation."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)

    # --- Data validation ---
    n_nan = int(df.isna().sum().sum())
    n_inf = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            finite_vals = df[col][np.isfinite(df[col])]
            if len(finite_vals) > 0:
                df[col] = df[col].clip(lower=finite_vals.min(), upper=finite_vals.max())

    if len(df.columns) != len(CMAPSS_COLUMNS):
        log.error(
            "Column count mismatch",
            extra={"expected": len(CMAPSS_COLUMNS), "got": len(df.columns)},
        )
        raise ValueError(
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


def load_rul(filepath: str | Path) -> pd.Series:
    """Load ground-truth RUL file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")
    rul = pd.read_csv(path, header=None, names=["RUL"])["RUL"]
    log.info("RUL loaded", extra={"units": len(rul)})
    return rul


def drop_noisy_columns(
    df: pd.DataFrame,
    drop_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Drop constant/noisy columns."""
    cols = drop_cols or DEFAULT_DROP_COLS
    df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
    log.info(
        "Dropped noisy columns",
        extra={"dropped": len(cols), "remaining": len(df.columns)},
    )
    return df


def ingest(
    train_path: str | Path,
    test_path: str | Path,
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
