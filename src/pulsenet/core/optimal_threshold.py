"""
PulseNet core utility for finding optimal anomaly thresholds.
"""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def find_optimal_threshold(
    y_true: np.ndarray | Any, 
    scores: np.ndarray | Any, 
    n_steps: int = 100
) -> float:
    """Find the threshold that maximizes F1 score."""
    best_f1 = 0.0
    best_threshold = 0.0
    
    # Range from min to max score
    min_score, max_score = float(np.min(scores)), float(np.max(scores))
    thresholds = np.linspace(min_score, max_score, n_steps)
    
    for t in thresholds:
        y_pred = (scores > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)  # type: ignore
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(t)
            
    return best_threshold
