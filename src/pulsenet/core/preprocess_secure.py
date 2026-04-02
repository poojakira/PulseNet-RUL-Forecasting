# pyright: reportGeneralTypeIssues=false
"""
Secure data preprocessing for PulseNet.
Uses EncryptionManager for enterprise-grade protection.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from pulsenet.logger import get_logger
from pulsenet.security.encryption import EncryptionManager

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Setup & Security
# ---------------------------------------------------------------------------
encryption = EncryptionManager()

# Standard C-MAPSS Column Names
COLS = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Noisy sensors to drop (specific to FD001)
DROP_COLS = [
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


def preprocess_data(
    train_path: str = "train_FD001.txt",
    test_path: str = "test_FD001.txt",
    output_dir: str = "data/preprocessed",
) -> None:
    """Load, clean, encrypt and save C-MAPSS datasets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        log.error(f"Source files {train_path} or {test_path} not found.")
        return

    log.info(f"Loading {train_path} and {test_path}")
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)

    # Drop noisy sensors
    train_df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    test_df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    log.info(
        f"Encrypting {train_df.shape[0]} training rows and {test_df.shape[0]} test rows"
    )

    # Use enterprise EncryptionManager
    train_enc = encryption.encrypt_dataframe(train_df)
    test_enc = encryption.encrypt_dataframe(test_df)

    train_out = out_path / "train_enc.csv"
    test_out = out_path / "test_enc.csv"

    train_enc.to_csv(train_out, index=False)
    test_enc.to_csv(test_out, index=False)

    log.info(f"Secure preprocessing complete. Saved to {out_path}")


if __name__ == "__main__":
    preprocess_data()
