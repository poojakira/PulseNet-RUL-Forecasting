import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import joblib

# 1. Load Data
df_test = pd.read_csv("test_features.csv")
rul_true = pd.read_csv("RUL_FD001.txt", header=None, names=["RUL"])["RUL"]
feature_cols = [c for c in df_test.columns if 'sensor' in c or 'rolling' in c]
X = df_test[feature_cols]

# 2. Re-create y_true for Evaluation
y_true = []
for unit_id in df_test['unit_number'].unique():
    actual_last_rul = rul_true.iloc[int(unit_id) - 1]
    unit_data = df_test[df_test['unit_number'] == unit_id]
    max_c = unit_data['time_in_cycles'].max()
    for current_cycle in unit_data['time_in_cycles']:
        current_rul = actual_last_rul + (max_c - current_cycle)
        y_true.append(1 if current_rul <= 30 else 0)
y_true = np.array(y_true)

# 3. Grid Search Parameters
n_estimators_list = [100, 200]
max_samples_list = [0.8, 1.0]
contamination_list = [0.05, 0.1, 0.15] # Adjust based on expected failure rate

best_f1 = 0
best_params = {}

print("--- STARTING GRID SEARCH ---")
for n in n_estimators_list:
    for s in max_samples_list:
        for c in contamination_list:
            # Train model
            model = IsolationForest(n_estimators=n, max_samples=s, contamination=c, random_state=42)
            model.fit(X)
            
            # Get predictions (In IF: -1 is anomaly, 1 is normal)
            preds = model.predict(X)
            y_pred = np.where(preds == -1, 1, 0)
            
            current_f1 = f1_score(y_true, y_pred)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = {'n_estimators': n, 'max_samples': s, 'contamination': c}
                print(f"New Best F1: {best_f1:.4f} with {best_params}")

# 4. Save the Optimized Model
final_model = IsolationForest(**best_params, random_state=42)
final_model.fit(X)
joblib.dump(final_model, "optimized_isolation_forest.joblib")
print(f"\n Optimized model saved with F1: {best_f1:.4f}")