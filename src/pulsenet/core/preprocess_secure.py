import pandas as pd
from cryptography.fernet import Fernet
import os

# 1. SETUP & SECURITY
# Generate a key and save it
key = Fernet.generate_key()
with open("secret.key", "wb") as f:
    f.write(key)
cipher = Fernet(key)

# 2. LOAD DATA
# Standard C-MAPSS Column Names
cols = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Check if files exist before reading
if not os.path.exists("train_FD001.txt") or not os.path.exists("test_FD001.txt"):
    print("Error: train_FD001.txt or test_FD001.txt not found. Please upload them.")
else:
    train_df = pd.read_csv("train_FD001.txt", sep="\s+", header=None, names=cols)
    test_df = pd.read_csv("test_FD001.txt", sep="\s+", header=None, names=cols)

    # 3. ADVANCED: DROP NOISY SENSORS (Specific to FD001)
    # These sensors are flat lines or noise and confuse the model
    drop_cols = [
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

    train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 4. ENCRYPT & SAVE (Simulating Secure Cloud Storage)
    print(
        f"Encrypting {train_df.shape[0]} training rows and {test_df.shape[0]} testing rows..."
    )

    # Helper to encrypt a dataframe
    def encrypt_df(df, output_name):
        # converting to string to encrypt, this is slow for large dfs but simulates the requirement
        df_enc = df.apply(
            lambda x: x.astype(str).apply(
                lambda val: cipher.encrypt(val.encode()).decode()
            )
        )
        df_enc.to_csv(output_name, index=False)

    encrypt_df(train_df, "preprocessed_train_enc.csv")
    encrypt_df(test_df, "preprocessed_test_enc.csv")

    print("Secure Preprocessing Complete. Encrypted files saved.")
