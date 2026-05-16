"""
Standalone ROC/AUC analysis script for trained Isolation Forest model.

Generates a ROC curve plot from the trained model + test features + ground truth RUL.

Usage:
    python -m pulsenet.evaluation.evaluate_roc

Requires:
    - Trained model: models/isolation_forest.joblib
    - Test features: data/test_features.csv
    - Ground truth: data/RUL_FD001.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from pulsenet.logger import get_logger
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.pipeline.preprocessing import create_labels, get_feature_columns

log = get_logger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
ROOT = Path(__file__).resolve().parents[3]
MODEL_FILE = ROOT / "models" / "isolation_forest.joblib"
TEST_FEATURES_FILE = ROOT / "data" / "test_features.csv"
GROUND_TRUTH_FILE = ROOT / "data" / "RUL_FD001.txt"
PLOT_FILE = ROOT / "outputs" / "roc_curve_analysis.png"
FAILURE_THRESHOLD = 30  # Standard C-MAPSS threshold


def main() -> None:
    """Run ROC analysis end-to-end."""
    log.info("Starting ROC/AUC analysis")

    # Validate inputs exist
    for path, name in [
        (MODEL_FILE, "trained model"),
        (TEST_FEATURES_FILE, "test features CSV"),
        (GROUND_TRUTH_FILE, "ground truth RUL"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Run 'python main_pipeline.py --mode full' first."
            )

    # Load model via the proper wrapper class (not raw joblib)
    log.info(f"Loading model from {MODEL_FILE}")
    model = IsolationForestModel()
    model.load(MODEL_FILE)

    # Load test features
    log.info(f"Loading test features from {TEST_FEATURES_FILE}")
    df_test = pd.read_csv(TEST_FEATURES_FILE)

    # Load ground truth RUL
    log.info(f"Loading RUL ground truth from {GROUND_TRUTH_FILE}")
    rul_truth = pd.read_csv(GROUND_TRUTH_FILE, header=None, names=["RUL"])["RUL"]

    # Get feature columns (drop metadata)
    feature_cols = get_feature_columns(df_test)
    X_test = df_test[feature_cols].to_numpy()

    # Create binary labels using shared label creation function
    y_true = create_labels(df_test, rul_truth, failure_threshold=FAILURE_THRESHOLD)

    # Compute scores (negate decision_function so higher = more anomalous)
    log.info("Computing anomaly scores")
    raw_scores = model.decision_function(X_test)
    y_scores = -raw_scores

    # Sanity check: at least 2 classes present
    if len(np.unique(y_true)) < 2:
        log.error(
            f"Only one class in y_true. Cannot compute ROC. "
            f"Try lowering FAILURE_THRESHOLD (current: {FAILURE_THRESHOLD})."
        )
        sys.exit(1)

    # Compute ROC and AUC
    fpr, tpr, _thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    log.info(f"ROC AUC: {roc_auc:.4f}")
    print(f"\nResults\n{'=' * 40}\nROC AUC: {roc_auc:.4f}\n{'=' * 40}")

    # Plot
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Isolation Forest on C-MAPSS FD001")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"ROC plot saved: {PLOT_FILE}")
    print(f"Plot saved: {PLOT_FILE}")


if __name__ == "__main__":
    main()
