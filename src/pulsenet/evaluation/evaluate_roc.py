import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FILE = "isolation_forest_model.joblib"
TEST_FEATURES_FILE = "test_features.csv"
GROUND_TRUTH_FILE = "RUL_FD001.txt"
PLOT_FILENAME = "roc_curve_analysis.png"


# 1. LOAD RESOURCES

print("--- STARTING ROC/AUC ANALYSIS ---")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file '{MODEL_FILE}' not found. Run train_secure.py first."
    )
if not os.path.exists(TEST_FEATURES_FILE):
    raise FileNotFoundError(
        f"Test features '{TEST_FEATURES_FILE}' not found. Run features_secure.py first."
    )
if not os.path.exists(GROUND_TRUTH_FILE):
    raise FileNotFoundError(
        f"Ground Truth file '{GROUND_TRUTH_FILE}' not found. Please upload it."
    )

# Load Model
print(f"Loading model: {MODEL_FILE}")
model = joblib.load(MODEL_FILE)

# Load Test Features
print(f"Loading test features: {TEST_FEATURES_FILE}")
df_test = pd.read_csv(TEST_FEATURES_FILE)

# Load Ground Truth (RUL)
print(f"Loading ground truth: {GROUND_TRUTH_FILE}")
rul_true = pd.read_csv(GROUND_TRUTH_FILE, header=None, names=["RUL"])


# 2. PREPARE DATA & LABELS

print("Mapping Ground Truth labels to Test Data...")

# Get feature columns (exclude metadata)
feature_cols = model.feature_names_in_
X_test = df_test[feature_cols]

# Calculate max cycle for each unit in test data
max_cycles = df_test.groupby("unit_number")["time_in_cycles"].max()

y_true = []

# Iterate through engines to assign binary labels
for unit_id in df_test["unit_number"].unique():
    # FIXED: Force unit_id to int because it might be a float
    idx = int(unit_id)

    # Get data for this engine
    unit_data = df_test[df_test["unit_number"] == unit_id]

    # Get the true final remaining life for this engine
    # We use (idx - 1) because DataFrame index starts at 0 but Unit IDs start at 1
    final_rul = rul_true.iloc[idx - 1]["RUL"]
    max_cycle = max_cycles[unit_id]

    # Calculate 'True' label for every row
    for current_cycle in unit_data["time_in_cycles"]:
        current_rul = final_rul + (max_cycle - current_cycle)
        # Label = 1 (Failure/Anomaly) if RUL <= 30 cycles
        # Label = 0 (Normal) if RUL > 30 cycles
        label = 1 if current_rul <= 30 else 0
        y_true.append(label)


# 3. CALCULATE SCORES

print("Calculating anomaly scores...")

# Isolation Forest returns: > 0 for Normal, < 0 for Anomaly
raw_scores = model.decision_function(X_test)

# Invert scores for ROC: Higher Score must mean Higher Anomaly Risk
y_scores = -raw_scores


# 4. COMPUTE ROC & AUC

print("Computing ROC curve metrics...")
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print("\n========================================")
print(" RESULTS")
print("========================================")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("========================================")


# 5. PLOT & SAVE

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (Isolation Forest)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.savefig(PLOT_FILENAME)
print(f"\n ROC Curve plot saved as '{PLOT_FILENAME}'")
