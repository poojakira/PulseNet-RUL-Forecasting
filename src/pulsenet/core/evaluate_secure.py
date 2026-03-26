import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

# 1. LOAD DATA
if not os.path.exists("train_features.csv") or not os.path.exists("test_features.csv"):
    print("Error: Feature files not found. Run features_secure.py first.")
else:
    print("Loading features...")
    df_train = pd.read_csv("train_features.csv")
    df_test = pd.read_csv("test_features.csv")

    # Define feature columns
    feature_cols = [
        c for c in df_train.columns if c not in ["unit_number", "time_in_cycles"]
    ]

    # 2. ADVANCED TRAINING STRATEGY (Semi-Supervised)
    # Train ONLY on "Healthy" data (First 50 cycles of each engine)
    # This teaches the Isolation Forest what a "Good" engine looks like.
    healthy_data = df_train[df_train["time_in_cycles"] <= 50][feature_cols]

    print(f"Training Isolation Forest on {len(healthy_data)} healthy samples...")
    # contamination='auto' or a small float like 0.05 usually works well for this
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(healthy_data)

    # Save the model
    joblib.dump(model, "isolation_forest_model.joblib")

    # 3. EVALUATE ON TEST SET (Where is test_FD001 used?)
    # We define "Anomaly" = "Degraded Engine State"
    print("Predicting on Test Data...")
    X_test = df_test[feature_cols]
    y_pred_raw = model.predict(X_test)

    # Convert: 1 (Normal) -> 0, -1 (Anomaly/Degraded) -> 1
    y_pred = [0 if p == 1 else 1 for p in y_pred_raw]

    # Attach predictions to test dataframe
    df_test["is_anomaly"] = y_pred

    # 4. SUMMARY METRICS
    total_samples = len(df_test)
    anomalies_detected = sum(y_pred)
    ratio = anomalies_detected / total_samples

    print("\n--- TEST SET EVALUATION ---")
    print(f"Total Test Samples: {total_samples}")
    print(f"Degraded States Detected: {anomalies_detected}")
    print(f"Anomaly Ratio: {ratio:.2%}")

    # Save results for your Dashboard
    # We create a simple summary CSV for the dashboard to read
    results = pd.DataFrame(
        {
            "total_samples": [total_samples],
            "anomalies": [anomalies_detected],
            "anomaly_ratio": [ratio],
        }
    )
    results.to_csv("evaluation_results.csv", index=False)

    # Optionally save the full test predictions if you want detailed analysis later
    df_test.to_csv("test_predictions.csv", index=False)
    print("Evaluation Complete. Results saved to evaluation_results.csv")
