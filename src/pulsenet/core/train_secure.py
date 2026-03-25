import pandas as pd
from cryptography.fernet import Fernet
from sklearn.ensemble import IsolationForest
import joblib

# Load key
with open("secret.key", "rb") as f:
    key = f.read()
cipher = Fernet(key)

# Load decrypted features
df_enc = pd.read_csv("train_features.csv")
X_train = df_enc.drop(columns=["unit_number", "time_in_cycles"])

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train)

# Save model
joblib.dump(model, "isolation_forest_model.joblib")
print("Training complete.")



