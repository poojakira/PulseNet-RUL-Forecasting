# pyre-ignore-all-errors
"""
Auto-training pipeline with model and dataset versioning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np  # pyre-ignore

try:
    import torch  # pyre-ignore
    import torch.distributed as dist  # pyre-ignore

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pulsenet.models.base import BaseAnomalyModel  # pyre-ignore
from pulsenet.models.registry import ModelRegistry  # pyre-ignore
from pulsenet.logger import get_logger  # pyre-ignore

log = get_logger(__name__)


class TrainingPipeline:
    """Automated training with versioned model/dataset artifacts."""

    def __init__(
        self, model_dir: str = "./models", registry: Optional[ModelRegistry] = None
    ):
        self.model_dir = Path(model_dir)
        self.registry = registry or ModelRegistry()

        self.is_distributed = False
        self.rank = 0
        self._init_distributed()

        # Only Rank 0 creates directories
        if self.rank == 0:
            self.model_dir.mkdir(exist_ok=True)

    def _init_distributed(self):
        """Initialize PyTorch DDP process group if running via torchrun."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            import os

            if "LOCAL_RANK" in os.environ:
                if not dist.is_initialized():
                    # NCCL backend is highly optimized for NVIDIA GPUs
                    dist.init_process_group(backend="nccl")
                self.is_distributed = True
                self.rank = dist.get_rank()
                torch.cuda.set_device(self.rank)
                log.info(f"Initialized DDP environment on Rank {self.rank}")

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
        try:
            model.train(X_train)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and TORCH_AVAILABLE:
                log.error(
                    f"GPU OOM on {model_name}. Attempting recovery and cache clear..."
                )
                torch.cuda.empty_cache()
            raise e

        train_time = time.perf_counter() - t0

        # Tune if applicable (IF model)
        if tune and y_true is not None and hasattr(model, "tune") and self.rank == 0:
            tune_result = model.tune(X_train, y_true)
            log.info("Tuning result", extra=tune_result)

        # Optimize threshold
        if (
            y_true is not None
            and hasattr(model, "optimize_threshold")
            and self.rank == 0
        ):
            model.optimize_threshold(X_train, y_true)

        # -----------------------------------------------------------------
        # Only Rank 0 saves models to prevent concurrent write collisions
        # -----------------------------------------------------------------
        if self.rank == 0:
            save_path = self.model_dir / f"{model_name}_v{version}.joblib"
            model.save(save_path)

            latest_path = self.model_dir / f"{model_name}.joblib"
            model.save(latest_path)
        else:
            save_path = self.model_dir / f"{model_name}_v{version}.joblib"

        result = {
            "model": model_name,
            "version": version,
            "train_time_sec": round(train_time),  # pyre-ignore
            "samples": len(X_train),
            "path": str(save_path),
        }
        if self.rank == 0:
            import yaml
            card_path = self.model_dir / f"{model_name}_model_card.yaml"
            model_card = {
                "name": f"PulseNet-{model_name}",
                "version": version,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "architecture": model_name,
                "training_samples": len(X_train),
                "author": "Pooja Kiran (ML Infrastructure)",
                "license": "Apache-2.0",
                "train_time_sec": round(train_time),
            }
            with card_path.open("w") as f:
                yaml.dump(model_card, f, default_flow_style=False)
            
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
