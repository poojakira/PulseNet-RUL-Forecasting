# pyright: reportGeneralTypeIssues=false
"""
Isolation Forest anomaly detection model with hyperparameter tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
from sklearn.metrics import f1_score, roc_curve

try:
    from cuml.ensemble import IsolationForest as cumlIForest  # type: ignore

    CUML_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import IsolationForest as sklearnIForest

    CUML_AVAILABLE = False

from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel

log = get_logger(__name__)


class IsolationForestModel(BaseAnomalyModel):
    """Isolation Forest with grid-search tuning and threshold optimization."""

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: Optional[int] = None,
        contamination: Optional[float] = None,
        max_samples: Optional[Union[float, str]] = None,
        random_state: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        from pulsenet.config import cfg
        
        # Pull from config if not explicitly provided
        conf = cfg.models.isolation_forest
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators or conf.n_estimators,
            "contamination": contamination or conf.contamination,
            "max_samples": max_samples or conf.max_samples,
            "random_state": random_state or conf.random_state,
        }
        self.threshold = threshold
        self.model: Any = None

    def _ensure_model(self) -> None:
        """Ensure model is trained or loaded."""
        if self.model is None:
            raise RuntimeError(f"Model {self.name} is not trained or loaded.")

    def train(self, X: np.ndarray, **kwargs: Any) -> None:
        """Train Isolation Forest on healthy data."""
        if CUML_AVAILABLE:
            self.model = cumlIForest(**self.params)  # type: ignore
        else:
            self.model = sklearnIForest(**self.params)  # type: ignore

        self.model.fit(X)
        backend = "cuml(GPU)" if CUML_AVAILABLE else "sklearn(CPU)"
        log.info(
            f"IsolationForest trained via {backend}",
            extra={"samples": len(X), **self.params},
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions: 0 = normal, 1 = anomaly."""
        self._ensure_model()
        if self.threshold is not None:
            scores = self.score(X)
            return (scores >= self.threshold).astype(int)

        raw: np.ndarray = self.model.predict(X)
        return np.where(raw == -1, 1, 0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores (negated decision function: higher = more anomalous)."""
        self._ensure_model()
        return -np.array(self.model.decision_function(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw decision function (positive = normal, negative = anomaly)."""
        self._ensure_model()
        return np.array(self.model.decision_function(X))

    def health_index(self, X: np.ndarray) -> np.ndarray:
        """Convert scores to 0-100 health index."""
        self._ensure_model()
        raw: np.ndarray = self.model.decision_function(X)
        return np.clip(((raw + 0.15) / 0.3) * 100, 0, 100)

    def save(self, path: Union[Path, str]) -> None:
        """Persist model, threshold and params to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "threshold": self.threshold, "params": self.params},
            path,
        )
        log.info("IsolationForest saved", extra={"path": str(path)})

    def load(self, path: Union[Path, str]) -> None:
        """Load model, threshold and params from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.threshold = data.get("threshold")
        self.params = data.get("params", self.params)
        log.info("IsolationForest loaded", extra={"path": str(path)})

    def tune(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        n_estimators_list: Optional[list[int]] = None,
        contamination_list: Optional[list[float]] = None,
        max_samples_list: Optional[list[Union[float, str]]] = None,
    ) -> dict[str, Any]:
        """Grid search for best hyperparameters. Returns best params + F1."""
        n_est_l = n_estimators_list or [100, 200, 300]
        cont_l = contamination_list or [0.05, 0.10, 0.15]
        max_s_l = max_samples_list or [0.8, 1.0]

        best_f1 = 0.0
        best_params: dict[str, Any] = {}

        for n in n_est_l:
            for c in cont_l:
                for s in max_s_l:
                    if CUML_AVAILABLE:
                        mdl = cumlIForest(n_estimators=n, contamination=c, max_samples=s, random_state=42)  # type: ignore
                    else:
                        mdl = sklearnIForest(n_estimators=n, contamination=c, max_samples=s, random_state=42)  # type: ignore

                    mdl.fit(X)
                    preds = np.where(mdl.predict(X) == -1, 1, 0)
                    f1 = float(f1_score(y_true, preds, zero_division=0))  # type: ignore
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

    def optimize_threshold(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Find optimal threshold using Youden's J statistic."""
        scores = self.score(X)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        j = tpr - fpr
        self.threshold = float(thresholds[np.argmax(j)])
        log.info("Threshold optimized", extra={"threshold": f"{self.threshold:.4f}"})
        return self.threshold
