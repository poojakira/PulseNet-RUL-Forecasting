"""
Isolation Forest anomaly detection model with hyperparameter tuning.
# pyre-ignore-all-errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np  # pyre-ignore[21]

try:
    from cuml.ensemble import IsolationForest  # pyre-ignore[21]

    CUML_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import IsolationForest

    CUML_AVAILABLE = False
from sklearn.metrics import f1_score

from pulsenet.models.base import BaseAnomalyModel
from pulsenet.logger import get_logger

log = get_logger(__name__)


class IsolationForestModel(BaseAnomalyModel):
    """Isolation Forest with grid-search tuning and threshold optimization."""

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_samples: float | str = 0.8,
        random_state: int = 42,
        threshold: Optional[float] = None,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": max_samples,
            "random_state": random_state,
        }
        self.threshold = threshold
        self.model: Optional[IsolationForest] = None

    def train(self, X: np.ndarray | Any, **kwargs) -> None:
        """Train Isolation Forest on healthy data."""
        self.model = IsolationForest(**self.params)
        self.model.fit(X)
        backend = "cuml(GPU)" if CUML_AVAILABLE else "sklearn(CPU)"
        log.info(
            f"IsolationForest trained via {backend}",
            extra={"samples": len(X), **self.params},
        )

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Binary predictions: 0 = normal, 1 = anomaly."""
        if self.threshold is not None:
            scores = self.score(X)
            return (scores >= self.threshold).astype(int)
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0)

    def score(self, X: np.ndarray | Any) -> np.ndarray:
        """Anomaly scores (negated decision function: higher = more anomalous)."""
        return -self.model.decision_function(X)

    def decision_function(self, X: np.ndarray | Any) -> np.ndarray:
        """Raw decision function (positive = normal, negative = anomaly)."""
        return self.model.decision_function(X)

    def health_index(self, X: np.ndarray | Any) -> np.ndarray:
        """Convert scores to 0-100 health index."""
        raw = self.model.decision_function(X)
        return np.clip(((raw + 0.15) / 0.3) * 100, 0, 100)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "threshold": self.threshold, "params": self.params},
            path,
        )
        log.info("IsolationForest saved", extra={"path": str(path)})

    def load(self, path: Path | str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.threshold = data.get("threshold")
        self.params = data.get("params", self.params)
        log.info("IsolationForest loaded", extra={"path": str(path)})

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    def tune(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        n_estimators_list: list[int] | None = None,
        contamination_list: list[float] | None = None,
        max_samples_list: list[float] | None = None,
    ) -> dict:
        """Grid search for best hyperparameters. Returns best params + F1."""
        n_estimators_list = n_estimators_list or [100, 200, 300]
        contamination_list = contamination_list or [0.05, 0.10, 0.15]
        max_samples_list = max_samples_list or [0.8, 1.0]

        best_f1 = 0.0
        best_params = {}

        for n in n_estimators_list:
            for c in contamination_list:
                for s in max_samples_list:
                    mdl = IsolationForest(
                        n_estimators=n, contamination=c, max_samples=s, random_state=42
                    )
                    mdl.fit(X)
                    preds = np.where(mdl.predict(X) == -1, 1, 0)
                    f1 = f1_score(y_true, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            "n_estimators": n,
                            "contamination": c,
                            "max_samples": s,
                        }

        # Retrain with best params
        self.params.update(best_params)
        self.train(X)
        log.info("Tuning complete", extra={"best_f1": f"{best_f1:.4f}", **best_params})
        return {"best_f1": best_f1, "best_params": best_params}

    # ------------------------------------------------------------------
    # Threshold optimization (Youden's J)
    # ------------------------------------------------------------------
    def optimize_threshold(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Find optimal threshold using Youden's J statistic."""
        from sklearn.metrics import roc_curve

        scores = self.score(X)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        j = tpr - fpr
        self.threshold = float(thresholds[np.argmax(j)])
        log.info("Threshold optimized", extra={"threshold": f"{self.threshold:.4f}"})
        return self.threshold
