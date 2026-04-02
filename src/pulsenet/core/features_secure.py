# pyright: reportGeneralTypeIssues=false
"""
Secure feature engineering for PulseNet.
Decrypts preprocessed data, applies rolling features and normalization.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pulsenet.logger import get_logger
from pulsenet.security.encryption import EncryptionManager

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Setup & Security
# ---------------------------------------------------------------------------
encryption = EncryptionManager()


def engineering_features(
    input_dir: str = "data/preprocessed",
    output_dir: str = "data/features",
    model_dir: str = "models",
) -> None:
    """Decrypt data, engineer features, scale, and save."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    mod_path = Path(model_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    train_file = in_path / "train_enc.csv"
    test_file = in_path / "test_enc.csv"

    if not train_file.exists() or not test_file.exists():
        log.error(f"Encrypted source files not found in {in_path}")
        return

    # 1. DECRYPT
    log.info("Decrypting training and test data...")
    df_train_enc = pd.read_csv(train_file)
    df_test_enc = pd.read_csv(test_file)

    df_train = encryption.decrypt_dataframe(df_train_enc).astype(float)
    df_test = encryption.decrypt_dataframe(df_test_enc).astype(float)

    # 2. FEATURE ENGINEERING
    sensor_cols = [c for c in df_train.columns if c.startswith("sensor")]
    log.info(f"Generating rolling features for {len(sensor_cols)} sensors")

    for col in sensor_cols:
        rolling_col = f"{col}_rolling"
        df_train[rolling_col] = df_train[col].rolling(window=5, min_periods=1).mean()
        df_test[rolling_col] = df_test[col].rolling(window=5, min_periods=1).mean()

    # 3. NORMALIZATION
    # Identify feature columns (everything except IDs)
    id_cols = ["unit_number", "time_in_cycles"]
    feat_cols = [c for c in df_train.columns if c not in id_cols]

    log.info(f"Applying MinMaxScaler to {len(feat_cols)} features")
    scaler = MinMaxScaler()
    df_train[feat_cols] = scaler.fit_transform(df_train[feat_cols])
    df_test[feat_cols] = scaler.transform(df_test[feat_cols])

    # 4. SAVE
    # Save the scaler for inference use
    scaler_out = mod_path / "scaler.joblib"
    joblib.dump(scaler, scaler_out)

    train_out = out_path / "train_features.csv"
    test_out = out_path / "test_features.csv"

    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    log.info(
        f"Feature engineering complete. Artifacts saved to {out_path} and {mod_path}"
    )


if __name__ == "__main__":
    engineering_features()
