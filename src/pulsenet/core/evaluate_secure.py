# pyright: reportGeneralTypeIssues=false
"""
Model training and evaluation for PulseNet.
Uses Isolation Forest to detect engine degradation.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from pulsenet.config import cfg
from pulsenet.logger import get_logger

log = get_logger(__name__)


def train_and_evaluate(
    feature_dir: str = "data/features",
    model_dir: str = "models",
    output_dir: str = "data/results",
) -> None:
    """Train Isolation Forest on healthy data and evaluate on test set."""
    feat_path = Path(feature_dir)
    mod_path = Path(model_dir)
    out_path = Path(output_dir)

    feat_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = feat_path / "train_features.csv"
    test_file = feat_path / "test_features.csv"

    if not train_file.exists() or not test_file.exists():
        log.error(f"Feature files not found in {feat_path}")
        return

    log.info(f"Loading features from {feat_path}")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # 1. DEFINE FEATURES
    id_cols = ["unit_number", "time_in_cycles"]
    feature_cols = [c for c in df_train.columns if c not in id_cols]

    # 2. TRAINING (Semi-Supervised)
    # Train ONLY on "Healthy" data (e.g., first 50 cycles)
    healthy_mask = df_train["time_in_cycles"] <= 50
    healthy_data = df_train[healthy_mask][feature_cols]

    log.info(f"Training Isolation Forest on {len(healthy_data)} healthy samples...")

    # Use config from cfg
    n_estimators = getattr(cfg.models, "n_estimators", 100)
    contamination = getattr(cfg.models, "contamination", 0.05)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,  # type: ignore
        random_state=42,
    )  # type: ignore
    model.fit(healthy_data)

    # Save model artifact
    model_out = mod_path / "isolation_forest.joblib"
    joblib.dump(model, model_out)
    log.info(f"Model saved to {model_out}")

    # 3. EVALUATE
    log.info("Predicting on test data...")
    X_test = df_test[feature_cols]
    y_pred_raw = model.predict(X_test)

    # Convert: 1 (Normal) -> 0, -1 (Anomaly) -> 1
    y_pred = [0 if p == 1 else 1 for p in y_pred_raw]
    df_test["is_anomaly"] = y_pred

    # 4. SUMMARY
    total_samples = len(df_test)
    anomalies_detected = sum(y_pred)
    ratio = anomalies_detected / total_samples if total_samples > 0 else 0.0

    log.info(
        f"Test Set Summary: {total_samples} samples, {anomalies_detected} anomalies ({ratio:.2%})"
    )

    # Save results
    results_df = pd.DataFrame(
        {
            "total_samples": [total_samples],
            "anomalies": [anomalies_detected],
            "anomaly_ratio": [ratio],
        }
    )
    results_df.to_csv(out_path / "evaluation_results.csv", index=False)
    df_test.to_csv(out_path / "test_predictions.csv", index=False)

    log.info(f"Evaluation complete. Results saved to {out_path}")


if __name__ == "__main__":
    train_and_evaluate()
