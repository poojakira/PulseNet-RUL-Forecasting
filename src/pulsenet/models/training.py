# pyright: reportGeneralTypeIssues=false
"""
Auto-training pipeline with model and dataset versioning.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

from pulsenet.config import cfg
from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel
from pulsenet.models.registry import ModelRegistry

log = get_logger(__name__)


class TrainingPipeline:
    """Automated training with versioned model/dataset artifacts."""

    def __init__(
        self,
        model_dir: Union[str, Path] = "./models",
        registry: Optional[ModelRegistry] = None,
    ):
        self.model_dir = Path(model_dir)
        self.registry = registry or ModelRegistry()

        self.is_distributed = False
        self.rank = 0
        self._init_distributed()

        # Only Rank 0 creates directories
        if self.rank == 0:
            self.model_dir.mkdir(exist_ok=True)

    def _init_distributed(self) -> None:
        """Initialize PyTorch DDP process group if running via torchrun."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if "LOCAL_RANK" in os.environ:
                if not dist.is_initialized():
                    # NCCL is Linux-only; use Gloo for Windows stability
                    backend = "nccl" if platform.system() != "Windows" else "gloo"
                    dist.init_process_group(backend=backend)
                    log.info(f"Initialized DDP environment (backend: {backend})")

                self.is_distributed = True
                self.rank = dist.get_rank()
                # Only set device if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.rank)
                    log.info(f"Process Rank {self.rank} bound to GPU {self.rank}")

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        tune: bool = False,
    ) -> dict[str, Any]:
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

        train_time = float(time.perf_counter() - t0)

        # Optional tuning (if model supports it)
        if tune and y_true is not None and hasattr(model, "tune") and self.rank == 0:
            tune_func = getattr(model, "tune")
            tune_result = tune_func(X_train, y_true)
            log.info("Tuning result", extra=tune_result)

        # Optional threshold optimization
        if (
            y_true is not None
            and hasattr(model, "optimize_threshold")
            and self.rank == 0
        ):
            optim_func = getattr(model, "optimize_threshold")
            optim_func(X_train, y_true)

        # -----------------------------------------------------------------
        # Only Rank 0 saves models to prevent concurrent write collisions
        # -----------------------------------------------------------------
        save_path = self.model_dir / f"{model_name}_v{version}.joblib"
        if self.rank == 0:
            model.save(save_path)
            latest_path = self.model_dir / f"{model_name}.joblib"
            model.save(latest_path)

        result: dict[str, Any] = {
            "model": model_name,
            "version": version,
            "train_time_sec": round(train_time),
            "samples": len(X_train),
            "path": str(save_path),
        }

        if self.rank == 0:
            try:
                import yaml

                card_path = self.model_dir / f"{model_name}_model_card.yaml"
                model_card = {
                    "name": f"{cfg.system.name}-{model_name}",
                    "version": version,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "architecture": model_name,
                    "training_samples": len(X_train),
                    "author": f"{cfg.system.name} Core Engine",
                    "license": "Apache-2.0",
                    "train_time_sec": round(train_time),
                }
                with card_path.open("w") as f:
                    yaml.dump(model_card, f, default_flow_style=False)

                log.info("Training complete", extra=result)
            except ImportError:
                log.warning("yaml not found, skipping model card generation")

        return result

    def train_all(
        self,
        X_train: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Train all registered models."""
        results: list[dict[str, Any]] = []
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
