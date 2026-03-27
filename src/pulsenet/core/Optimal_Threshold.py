import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
MODEL_FILE = "isolation_forest_model.joblib"
TEST_FEATURES_FILE = "test_features.csv"
GROUND_TRUTH_FILE = "RUL_FD001.txt"

print("--- 1/4: LOADING RESOURCES ---")
if not all(
    os.path.exists(f) for f in [MODEL_FILE, TEST_FEATURES_FILE, GROUND_TRUTH_FILE]
):
    raise FileNotFoundError(
        "Ensure model, test_features, and RUL_FD001.txt are in the folder."
    )

model = joblib.load(MODEL_FILE)
df_test = pd.read_csv(TEST_FEATURES_FILE)
rul_true = pd.read_csv(GROUND_TRUTH_FILE, header=None, names=["RUL"])["RUL"]

# ==========================================
# 2. PREPARE DATA & LABELS
# ==========================================
print("--- 2/4: MAPPING GROUND TRUTH & SCORING ---")
feature_cols = model.feature_names_in_
X_test = df_test[feature_cols]
max_cycles = df_test.groupby("unit_number")["time_in_cycles"].max()

y_true = []
for unit_id in df_test["unit_number"].unique():
    idx = int(unit_id)
    unit_data = df_test[df_test["unit_number"] == unit_id]
    # Unit IDs start at 1, DataFrame index starts at 0
    actual_last_rul = rul_true.iloc[idx - 1]
    max_c = max_cycles[unit_id]

    for current_cycle in unit_data["time_in_cycles"]:
        current_rul = actual_last_rul + (max_c - current_cycle)
        y_true.append(1 if current_rul <= 30 else 0)

y_true = np.array(y_true)
# Negate scores so that LARGER values = Higher Anomaly Risk
y_scores = -model.decision_function(X_test)

# ==========================================
# 3. PERFORMANCE & THRESHOLD ANALYSIS
# ==========================================
print("--- 3/4: CALCULATING OPTIMAL THRESHOLD ---")
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Find Optimal Threshold using Youden's J-Statistic
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
y_pred = (y_scores >= best_threshold).astype(int)

# ==========================================
# 4. FEATURE IMPORTANCE (Permutation)
# ==========================================
print("--- 4/4: ANALYZING SENSOR IMPORTANCE ---")
X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
baseline_score = model.decision_function(X_sample).mean()
importances = []

for col in feature_cols:
    X_temp = X_sample.copy()
    X_temp[col] = np.random.permutation(X_temp[col].values)
    new_score = model.decision_function(X_temp).mean()
    importances.append(abs(baseline_score - new_score))

feat_importances = pd.Series(importances, index=feature_cols)

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: ROC Curve
axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
axes[0].plot([0, 1], [0, 1], color="navy", linestyle="--")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap="Blues", ax=axes[1], colorbar=False)
axes[1].set_title(f"CM at Threshold: {best_threshold:.3f}")

# Plot 3: Feature Importance
feat_importances.nlargest(10).plot(kind="barh", ax=axes[2], color="teal")
axes[2].set_title("Top 10 Sensor Contributors")

plt.tight_layout()
plt.savefig("complete_analysis_report.png")
plt.show()

print("\n========================================")
print(" FINAL RESULTS")
print("========================================")
print(f"ROC AUC:          {roc_auc:.4f}")
print(f"F1-Score:         {f1_score(y_true, y_pred):.4f}")
print(f"Best Threshold:   {best_threshold:.4f}")
print("========================================")
