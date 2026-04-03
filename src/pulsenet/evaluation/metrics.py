from __future__ import annotations

import numpy as np  # pyre-ignore
import pandas as pd  # pyre-ignore
from sklearn.metrics import (  # pyre-ignore
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from pulsenet.logger import get_logger  # pyre-ignore

log = get_logger(__name__)


def calculate_detection_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5
) -> dict:
    """Calculate standard classification metrics: Precision, Recall, F1, AUC-ROC, AUC-PR."""
    y_pred = (y_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Handle single-class case for ROC/PR
    if len(np.unique(y_true)) < 2:
        log.warning("Only one class present in y_true. ROC AUC and PR AUC are not defined.")
        roc_auc = 0.0
        pr_auc = 0.0
        avg_precision = 0.0
    else:
        roc_auc = roc_auc_score(y_true, y_scores)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(rec, prec)
        avg_precision = average_precision_score(y_true, y_scores)

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "avg_precision": round(float(avg_precision), 4),
    }


def calculate_lead_time(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    rul_truth: pd.Series,
    failure_threshold_cycles: int = 30,
) -> dict:
    """
    Calculate Lead Time: average cycles from first failure prediction to actual failure (RUL=0).
    A positive prediction for failure is defined when y_pred is 1.
    """
    lead_times = []
    engine_ids = df_test["unit_number"].unique()

    # Pre-calculate max cycles for efficiency
    max_cycles = df_test.groupby("unit_number")["time_in_cycles"].max()

    for i, unit_id in enumerate(engine_ids):
        # Indices for this engine in df_test
        mask = df_test["unit_number"] == unit_id
        indices = np.where(mask)[0]
        
        # Predicted labels for this engine
        unit_preds = y_pred[indices]
        
        # Check if model predicted failure for this engine
        if np.any(unit_preds == 1):
            # First cycle index where prediction was positive
            first_fail_idx = np.where(unit_preds == 1)[0][0]
            
            # Absolute cycle number for first prediction
            t_pred = df_test.iloc[indices[first_fail_idx]]["time_in_cycles"]
            
            # Actual final failure cycle in test data
            max_cycle = max_cycles[unit_id]
            
            # Ground truth RUL at the end of the test series for this engine
            # Unit IDs are 1-based, Series index is 0-based
            try:
                final_rul_true = rul_truth.iloc[int(unit_id) - 1]
            except IndexError:
                continue # Skip if no ground truth for this unit
                
            # Actual RUL at the time of prediction
            # RUL(t) = Final_RUL + (Max_Cycle - Current_Cycle)
            actual_rul_at_pred = final_rul_true + (max_cycle - t_pred)
            
            lead_times.append(actual_rul_at_pred)

    if not lead_times:
        return {"avg_lead_time": 0.0, "engines_detected": 0}

    return {
        "avg_lead_time": round(float(np.mean(lead_times)), 2),
        "median_lead_time": round(float(np.median(lead_times)), 2),
        "engines_detected": len(lead_times),
        "total_engines": len(engine_ids),
        "detection_rate": round(len(lead_times) / len(engine_ids), 4),
    }


def map_ground_truth_labels(
    df_test: pd.DataFrame, rul_truth: pd.Series, threshold_cycles: int = 30
) -> np.ndarray:
    """Assign binary labels based on ground truth RUL values."""
    y_true = []
    max_cycles = df_test.groupby("unit_number")["time_in_cycles"].max()
    
    for _, row in df_test.iterrows():
        unit_id = row["unit_number"]
        current_cycle = row["time_in_cycles"]
        
        final_rul = rul_truth.iloc[int(unit_id) - 1]
        max_cycle = max_cycles[unit_id]
        
        current_rul = final_rul + (max_cycle - current_cycle)
        label = 1 if current_rul <= threshold_cycles else 0
        y_true.append(label)
        
    return np.array(y_true)
