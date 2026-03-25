"""
Pipeline orchestrator — coordinates ingestion → preprocess → train → evaluate → log.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from pulsenet.config import cfg
from pulsenet.logger import get_logger
from pulsenet.pipeline.ingestion import ingest, load_rul
from pulsenet.pipeline.preprocessing import (
    preprocess_pipeline,
    get_feature_columns,
    create_labels,
)
from pulsenet.security.encryption import EncryptionManager
from pulsenet.security.blockchain import BlackBoxLedger
from pulsenet.models.registry import ModelRegistry

log = get_logger(__name__)


class PipelineOrchestrator:
    """Central pipeline controller for end-to-end execution."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.encryption = EncryptionManager()
        self.ledger = BlackBoxLedger()
        self.registry = ModelRegistry()
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.rul: Optional[pd.Series] = None

    def run_ingestion(self) -> None:
        """Stage 1: Load and clean data."""
        log.info("Stage 1 — Ingestion")
        train_path = self.data_dir / getattr(cfg.data, "train_file", "train_FD001.txt")
        test_path = self.data_dir / getattr(cfg.data, "test_file", "test_FD001.txt")
        rul_path = self.data_dir / getattr(cfg.data, "rul_file", "RUL_FD001.txt")

        self.train_df, self.test_df = ingest(train_path, test_path)
        self.rul = load_rul(rul_path)

    def run_preprocessing(self) -> None:
        """Stage 2: Feature engineering + normalization."""
        log.info("Stage 2 — Preprocessing")
        assert self.train_df is not None, "Run ingestion first"
        window = getattr(cfg.data, "rolling_window", 5)
        self.train_df, self.test_df, self.scaler = preprocess_pipeline(
            self.train_df, self.test_df, rolling_window=window
        )
        # Save features for downstream
        self.train_df.to_csv(self.data_dir / "train_features.csv", index=False)
        self.test_df.to_csv(self.data_dir / "test_features.csv", index=False)
        log.info("Features saved")

    def run_training(self, model_name: Optional[str] = None) -> None:
        """Stage 3: Train model(s)."""
        log.info("Stage 3 — Training")
        model_name = model_name or getattr(cfg.models, "active_model", "isolation_forest")
        feat_cols = get_feature_columns(self.train_df)

        healthy_limit = getattr(cfg.data, "healthy_cycle_limit", 50)
        healthy_data = self.train_df[self.train_df["time_in_cycles"] <= healthy_limit][feat_cols]

        model = self.registry.get_model(model_name)
        t0 = time.perf_counter()
        model.train(healthy_data)
        train_time = time.perf_counter() - t0
        log.info(f"Model '{model_name}' trained",
                 extra={"samples": len(healthy_data), "time_sec": f"{train_time:.2f}"})

        # Save model
        model_dir = Path(getattr(cfg.models, "model_dir", "./models"))
        model_dir.mkdir(exist_ok=True)
        model.save(model_dir / f"{model_name}.joblib")

    def run_evaluation(self) -> dict:
        """Stage 4: Evaluate on test set."""
        log.info("Stage 4 — Evaluation")
        feat_cols = get_feature_columns(self.test_df)
        threshold = getattr(cfg.data, "failure_rul_threshold", 30)
        y_true = create_labels(self.test_df, self.rul, failure_threshold=threshold)

        results = self.registry.compare_all(
            self.test_df[feat_cols].values, y_true
        )
        log.info("Evaluation complete", extra={"models": len(results)})
        return results

    def run_inference(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Stage 5: Run inference on test set, log to blockchain."""
        log.info("Stage 5 — Inference + Logging")
        model_name = model_name or getattr(cfg.models, "active_model", "isolation_forest")
        model = self.registry.get_model(model_name)
        feat_cols = get_feature_columns(self.test_df)

        predictions = model.predict(self.test_df[feat_cols])
        self.test_df["prediction"] = predictions

        # Log to blockchain (sample every 50th row)
        for idx in range(0, len(self.test_df), 50):
            row = self.test_df.iloc[idx]
            self.ledger.add_entry(
                unit_id=int(row["unit_number"]),
                cycles=int(row["time_in_cycles"]),
                health_score=float(1 - predictions[idx]) * 100,
                status="CRITICAL" if predictions[idx] == 1 else "OPTIMAL",
            )

        log.info("Inference complete",
                 extra={"anomalies": int(predictions.sum()), "total": len(predictions)})
        return self.test_df

    def run_full_pipeline(self) -> dict:
        """Execute all stages end-to-end."""
        log.info("=== PulseNet Full Pipeline Start ===")
        t0 = time.perf_counter()

        self.run_ingestion()
        self.run_preprocessing()
        self.run_training()
        results = self.run_evaluation()
        self.run_inference()

        duration = time.perf_counter() - t0
        log.info("=== Pipeline Complete ===",
                 extra={"duration_sec": f"{duration:.2f}"})
        return results
