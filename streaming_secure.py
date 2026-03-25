import pandas as pd
from cryptography.fernet import Fernet
import joblib
import time
import matplotlib.pyplot as plt


# Step 1: Load encryption key

with open("secret.key", "rb") as f:
    key = f.read()
cipher = Fernet(key)


# Step 2: Load model

model = joblib.load("isolation_forest_model.joblib")
feature_names = list(model.feature_names_in_)


# Step 3: Load encrypted data

df_enc = pd.read_csv("preprocessed_test_enc.csv")


# Step 4: Decryption helper

def decrypt_cell(val):
    try:
        return float(cipher.decrypt(val.encode()).decode())
    except:
        return 0.0

encrypted_features = df_enc.columns.tolist()


# Step 5: Rolling window

rolling_window = 3
prev_rows = []
results = []


# Step 6: Streaming inference

for idx, row in df_enc.iterrows():
    row_dec = row.copy()

    # Decrypt ALL columns 
    for col in encrypted_features:
        row_dec[col] = decrypt_cell(row_dec[col])

    #  Rolling window 
    prev_rows.append(row_dec)
    if len(prev_rows) > rolling_window:
        prev_rows.pop(0)

    #  Rolling mean features 
    for i in range(1, 22):
        s = f"sensor_{i}"
        r = f"sensor_{i}_rolling_mean"
        values = [pr[s] for pr in prev_rows]
        row_dec[r] = sum(values) / len(values)

    #  Build feature-aligned row 
    X_row_df = pd.DataFrame(
        [[row_dec.get(f, 0.0) for f in feature_names]],
        columns=feature_names
    )

    #  Decision score 
    score = model.decision_function(X_row_df)[0]

    # Sensitivity logic
    # Normal IF clearly positive
    # Borderline IF near zero
    # Anomaly IF negative
    if score < -0.05:
        label = "Anomaly"
    elif score < 0.02:
        label = "Borderline"
    else:
        label = "Normal"

    results.append({
        "cycle": int(row_dec["time_in_cycles"]),
        "score": score,
        "label": label
    })

    print(
        f"Row {idx} | Cycle {int(row_dec['time_in_cycles'])} "
        f"| Score {score:.4f} | {label}"
    )

    time.sleep(0.03)

# Step 7: Visualization

results_df = pd.DataFrame(results)

color_map = {
    "Normal": "green",
    "Borderline": "orange",
    "Anomaly": "red"
}

plt.figure(figsize=(16, 4))
plt.scatter(
    results_df["cycle"],
    results_df["score"],
    c=results_df["label"].map(color_map),
    s=18
)

plt.axhline(0, color="white", linestyle="--", linewidth=1)
plt.xlabel("Time in Cycles")
plt.ylabel("Anomaly Score (decision_function)")
plt.title("Isolation Forest Anomaly Score Timeline")
plt.grid(True)
plt.tight_layout()
plt.savefig("anomaly_score_timeline.png")
plt.show()

print("\nSaved plot as anomaly_score_timeline.png")
