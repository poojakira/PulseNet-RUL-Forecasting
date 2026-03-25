import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# 1. Load Data and Models
df_test = pd.read_csv("test_features.csv")
rul_true = pd.read_csv("RUL_FD001.txt", header=None, names=["RUL"])["RUL"]
original_model = joblib.load("isolation_forest_model.joblib")
optimized_model = joblib.load("optimized_isolation_forest.joblib")

feature_cols = original_model.feature_names_in_
X = df_test[feature_cols]

# 2. Re-create y_true
y_true = []
for unit_id in df_test['unit_number'].unique():
    actual_last_rul = rul_true.iloc[int(unit_id) - 1]
    unit_data = df_test[df_test['unit_number'] == unit_id]
    max_c = unit_data['time_in_cycles'].max()
    for current_cycle in unit_data['time_in_cycles']:
        current_rul = actual_last_rul + (max_c - current_cycle)
        y_true.append(1 if current_rul <= 30 else 0)
y_true = np.array(y_true)

# 3. Get Predictions
# Original (Using the best threshold you found: -0.0097)
orig_scores = -original_model.decision_function(X)
y_pred_orig = (orig_scores >= -0.0097).astype(int)

# Optimized (Using built-in predict from Grid Search)
y_pred_opt = np.where(optimized_model.predict(X) == -1, 1, 0)

# 4. Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_true, y_pred_orig, ax=axes[0], cmap='Reds', colorbar=False)
axes[0].set_title(f"Original Model\nF1: {f1_score(y_true, y_pred_orig):.4f}")

ConfusionMatrixDisplay.from_predictions(y_true, y_pred_opt, ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title(f"Optimized Model\nF1: {f1_score(y_true, y_pred_opt):.4f}")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

print(f"Comparison plot saved as 'model_comparison.png'")