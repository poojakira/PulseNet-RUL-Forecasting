"""
Auto-training pipeline with model and dataset versioning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from pulsenet.models.base import BaseAnomalyModel
from pulsenet.models.registry import ModelRegistry
from pulsenet.logger import get_logger

log = get_logger(__name__)


class TrainingPipeline:
    """Automated training with versioned model/dataset artifacts."""

    def __init__(self, model_dir: str = "./models", registry: Optional[ModelRegistry] = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.registry = registry or ModelRegistry()

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        tune: bool = False,
    ) -> dict:
        """Train a single model, optionally tune, save versioned artifact."""
        model = self.registry.get_model(model_name)
        version = time.strftime("%Y%m%d_%H%M%S")

        t0 = time.perf_counter()
        model.train(X_train)
        train_time = time.perf_counter() - t0

        # Tune if applicable (IF model)
        if tune and y_true is not None and hasattr(model, "tune"):
            tune_result = model.tune(X_train, y_true)
            log.info("Tuning result", extra=tune_result)

        # Optimize threshold
        if y_true is not None and hasattr(model, "optimize_threshold"):
            model.optimize_threshold(X_train, y_true)

        # Save versioned
        save_path = self.model_dir / f"{model_name}_v{version}.joblib"
        model.save(save_path)

        # Also save as "latest"
        latest_path = self.model_dir / f"{model_name}.joblib"
        model.save(latest_path)

        result = {
            "model": model_name,
            "version": version,
            "train_time_sec": round(train_time, 2),
            "samples": len(X_train),
            "path": str(save_path),
        }
        log.info("Training complete", extra=result)
        return result

    def train_all(
        self,
        X_train: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Train all registered models."""
        results = []
        for name in self.registry.available_models:
            try:
                r = self.train_model(name, X_train, y_true)
                results.append(r)
            except Exception as e:
                log.warning(f"Training failed for {name}: {e}")
                results.append({"model": name, "error": str(e)})
        return results

    def load_latest(self, model_name: str) -> BaseAnomalyModel:
        """Load the latest saved model."""
        path = self.model_dir / f"{model_name}.joblib"
        model = self.registry.get_model(model_name)
        model.load(path)
        return model
