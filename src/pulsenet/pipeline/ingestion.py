"""
Data ingestion — loads NASA C-MAPSS data and applies AES-256 encryption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pulsenet.logger import get_logger

log = get_logger(__name__)

# Standard C-MAPSS column names
CMAPSS_COLUMNS = (
    ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors known to be flat / noisy in FD001
DEFAULT_DROP_COLS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]


def load_raw(filepath: str | Path) -> pd.DataFrame:
    """Load raw C-MAPSS whitespace-separated file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    log.info("Raw data loaded", extra={"file": path.name, "rows": len(df)})
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
    log.info("Dropped noisy columns", extra={"dropped": len(cols), "remaining": len(df.columns)})
    return df


def ingest(
    train_path: str | Path,
    test_path: str | Path,
    drop_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full ingestion: load + clean both train and test sets."""
    train_df = drop_noisy_columns(load_raw(train_path), drop_cols)
    test_df = drop_noisy_columns(load_raw(test_path), drop_cols)
    log.info("Ingestion complete",
             extra={"train_rows": len(train_df), "test_rows": len(test_df)})
    return train_df, test_df
