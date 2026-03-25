"""
Abstract base class for all PulseNet anomaly detection models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class BaseAnomalyModel(ABC):
    """Interface that all anomaly detection models must implement."""

    name: str = "base"

    @abstractmethod
    def train(self, X: np.ndarray | Any, **kwargs) -> None:
        """Train the model on (healthy) data."""

    @abstractmethod
    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Return binary predictions: 0 = normal, 1 = anomaly."""

    @abstractmethod
    def score(self, X: np.ndarray | Any) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load model from disk."""

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Compute standard classification metrics."""
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, roc_auc_score,
        )

        y_pred = self.predict(X)
        scores = self.score(X)

        metrics: dict[str, float] = {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["roc_auc"] = 0.0

        return metrics
