import pandas as pd
from cryptography.fernet import Fernet
from sklearn.preprocessing import MinMaxScaler
import os

# 1. LOAD SECURITY KEY
if not os.path.exists("secret.key"):
    print("Error: secret.key not found. Run preprocess_secure.py first.")
else:
    with open("secret.key", "rb") as f:
        key = f.read()
    cipher = Fernet(key)

    # 2. DECRYPT DATA
    def decrypt_df(filename):
        if not os.path.exists(filename):
            print(f"Error: {filename} not found.")
            return pd.DataFrame()

        df_enc = pd.read_csv(filename)
        # Decrypt every cell
        return df_enc.apply(
            lambda x: x.astype(str).apply(
                lambda val: cipher.decrypt(val.encode()).decode()
            )
        )

    print("Decrypting data (this may take a moment)...")
    df_train = decrypt_df("preprocessed_train_enc.csv")
    df_test = decrypt_df("preprocessed_test_enc.csv")

    if not df_train.empty and not df_test.empty:
        # Convert to float
        df_train = df_train.astype(float)
        df_test = df_test.astype(float)

        # 3. ADVANCED FEATURE ENGINEERING
        # Identify sensor columns (those that are left after dropping noise)
        sensor_cols = [c for c in df_train.columns if c.startswith("sensor")]

        # Rolling Mean (Smoothing)
        print("Generating Rolling Features...")
        for col in sensor_cols:
            df_train[f"{col}_rolling"] = (
                df_train[col].rolling(window=5, min_periods=1).mean()
            )
            df_test[f"{col}_rolling"] = (
                df_test[col].rolling(window=5, min_periods=1).mean()
            )

        # 4. NORMALIZATION (CRITICAL STEP)
        # We fit the scaler on TRAIN, but apply it to TEST.
        scaler = MinMaxScaler()

        # Select all feature columns (excluding IDs)
        feat_cols = [
            c for c in df_train.columns if c not in ["unit_number", "time_in_cycles"]
        ]

        df_train[feat_cols] = scaler.fit_transform(df_train[feat_cols])
        df_test[feat_cols] = scaler.transform(df_test[feat_cols])

        # 5. SAVE FEATURES
        df_train.to_csv("train_features.csv", index=False)
        df_test.to_csv("test_features.csv", index=False)

        print("Feature Engineering Complete. Scaled Train and Test files saved.")
