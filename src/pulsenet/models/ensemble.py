"""
Ensemble model — combines multiple anomaly detectors via majority voting or weighted scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel

log = get_logger(__name__)


class EnsembleModel(BaseAnomalyModel):
    """Ensemble that combines Isolation Forest, LSTM, and Transformer via majority vote.

    Strategies:
        - "majority_vote": Binary vote across sub-models (default)
        - "weighted_score": Weighted average of anomaly scores
    """

    name = "ensemble"

    def __init__(
        self,
        strategy: str = "majority_vote",
        weights: Optional[list[float]] = None,
        threshold: float = 0.5,
    ):
        self.strategy = strategy
        self.weights = weights
        self.threshold = threshold
        self._sub_models: list[BaseAnomalyModel] = []
        self._model_names: list[str] = []

    def _load_sub_models(self) -> None:
        """Lazy-import and instantiate sub-models."""
        if self._sub_models:
            return

        from pulsenet.models.isolation_forest import IsolationForestModel
        from pulsenet.models.lstm_model import LSTMModel
        from pulsenet.models.transformer_model import TransformerModel

        self._sub_models = [
            IsolationForestModel(),
            LSTMModel(),
            TransformerModel(),
        ]
        self._model_names = [m.name for m in self._sub_models]

        if self.weights is None:
            self.weights = [1.0 / len(self._sub_models)] * len(self._sub_models)

        log.info("Ensemble sub-models loaded", extra={"models": self._model_names})

    def train(self, X: np.ndarray | Any, **kwargs) -> None:
        """Train all sub-models on the same healthy data."""
        self._load_sub_models()
        for model in self._sub_models:
            log.info(f"Training ensemble member: {model.name}")
            model.train(X, **kwargs)
        log.info("Ensemble training complete", extra={"members": len(self._sub_models)})

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict anomalies using the configured ensemble strategy."""
        if self.strategy == "weighted_score":
            scores = self.score(X)
            return (scores >= self.threshold).astype(int)
        return self._majority_vote(X)

    def _majority_vote(self, X: np.ndarray | Any) -> np.ndarray:
        """Majority voting: anomaly if >50% of models flag it."""
        votes = np.zeros(len(X), dtype=int)
        for model in self._sub_models:
            votes += model.predict(X)
        majority = len(self._sub_models) / 2.0
        return (votes > majority).astype(int)

    def score(self, X: np.ndarray | Any) -> np.ndarray:
        """Weighted average of anomaly scores from all sub-models."""
        all_scores = []
        for model in self._sub_models:
            s = model.score(X)
            # Normalize to [0, 1] range per model
            s_min, s_max = s.min(), s.max()
            if s_max - s_min > 0:
                s = (s - s_min) / (s_max - s_min)
            all_scores.append(s)

        stacked = np.column_stack(all_scores)  # (N, num_models)
        return np.average(stacked, axis=1, weights=self.weights)

    def save(self, path: Path | str) -> None:
        """Save all sub-models and ensemble config."""
        path = Path(path)
        ensemble_dir = path.parent / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        for model in self._sub_models:
            model.save(ensemble_dir / f"{model.name}.joblib")

        config = {
            "strategy": self.strategy,
            "weights": self.weights,
            "threshold": self.threshold,
            "model_names": self._model_names,
        }
        with open(ensemble_dir / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info("Ensemble saved", extra={"path": str(ensemble_dir)})

    def load(self, path: Path | str) -> None:
        """Load all sub-models and ensemble config."""
        path = Path(path)
        ensemble_dir = path.parent / "ensemble"

        config_path = ensemble_dir / "ensemble_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.strategy = config.get("strategy", self.strategy)
            self.weights = config.get("weights", self.weights)
            self.threshold = config.get("threshold", self.threshold)

        self._load_sub_models()
        for model in self._sub_models:
            model_path = ensemble_dir / f"{model.name}.joblib"
            if model_path.exists():
                model.load(model_path)
                log.info(f"Loaded ensemble member: {model.name}")

        log.info("Ensemble loaded", extra={"path": str(ensemble_dir)})
