"""
Multi-model registry — manages model instances and comparison.
"""

from __future__ import annotations

import numpy as np

from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel
from pulsenet.models.isolation_forest import IsolationForestModel

log = get_logger(__name__)


class ModelRegistry:
    """Registry for managing and comparing multiple anomaly detection models."""

    def __init__(self):
        self._models: dict[str, BaseAnomalyModel] = {}
        # Always register Isolation Forest
        self.register(IsolationForestModel())

    def register(self, model: BaseAnomalyModel) -> None:
        """Register a model instance."""
        self._models[model.name] = model
        log.info(f"Model registered: {model.name}")

    def get_model(self, name: str) -> BaseAnomalyModel:
        """Get a registered model by name."""
        if name not in self._models:
            # Lazy-load PyTorch models
            if name == "lstm":
                from pulsenet.models.lstm_model import LSTMModel

                self.register(LSTMModel())
            elif name == "transformer":
                from pulsenet.models.transformer_model import TransformerModel

                self.register(TransformerModel())
            elif name == "ensemble":
                from pulsenet.models.ensemble import EnsembleModel

                self.register(EnsembleModel())
            else:
                raise KeyError(
                    f"Unknown model: {name}. Available: {list(self._models.keys())}"
                )
        return self._models[name]

    @property
    def available_models(self) -> list[str]:
        return list(self._models.keys())

    def compare_all(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, dict]:
        """Evaluate all registered (trained) models and return comparison."""
        results = {}
        for name, model in self._models.items():
            try:
                metrics = model.evaluate(X, y_true)
                results[name] = metrics
                log.info(
                    f"Model '{name}' evaluation",
                    extra={k: f"{v:.4f}" for k, v in metrics.items()},
                )
            except Exception as e:
                log.warning(f"Model '{name}' evaluation failed: {e}")
                results[name] = {"error": str(e)}
        return results

    def best_model(self, X: np.ndarray, y_true: np.ndarray, metric: str = "f1") -> str:
        """Return name of the best model by given metric."""
        results = self.compare_all(X, y_true)
        best_name = max(
            (name for name in results if "error" not in results[name]),
            key=lambda n: results[n].get(metric, 0),
            default="isolation_forest",
        )
        log.info(f"Best model by {metric}: {best_name}")
        return best_name
