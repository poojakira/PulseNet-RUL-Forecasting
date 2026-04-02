# pyright: reportGeneralTypeIssues=false
"""
Pipeline orchestrator — coordinates ingestion → preprocess → train → evaluate → log.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, cast

import joblib
import pandas as pd

from pulsenet.config import cfg
from pulsenet.core.exceptions import DataError, ModelError, PulseNetError
from pulsenet.logger import get_logger
from pulsenet.models.registry import ModelRegistry
from pulsenet.pipeline.ingestion import ingest, load_rul
from pulsenet.pipeline.preprocessing import (create_labels, create_sequences,
                                             get_feature_columns,
                                             preprocess_pipeline)
from pulsenet.security.blockchain import BlackBoxLedger

log = get_logger(__name__)


class PipelineOrchestrator:
    """Central pipeline controller for end-to-end execution."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.ledger = BlackBoxLedger()
        self.registry = ModelRegistry()
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.rul: Optional[pd.Series] = None
        self.scaler: Any = None

    def run_ingestion(self) -> None:
        """Stage 1: Load and clean data."""
        try:
            log.info("Stage 1 — Ingestion")
            train_path = self.data_dir / cfg.data.train_file
            test_path = self.data_dir / cfg.data.test_file
            rul_path = self.data_dir / cfg.data.rul_file

            if not train_path.exists():
                raise FileNotFoundError(f"Data file not found: {train_path}")

            self.train_df, self.test_df = ingest(train_path, test_path)
            self.rul = load_rul(rul_path)
        except Exception as e:
            raise DataError(f"Ingestion failed: {e}") from e

    def run_preprocessing(self) -> None:
        """Stage 2: Feature engineering + normalization."""
        try:
            log.info("Stage 2 — Preprocessing")
            if self.train_df is None or self.test_df is None:
                raise DataError("Run ingestion first")

            window = cfg.data.rolling_window
            self.train_df, self.test_df, self.scaler = preprocess_pipeline(
                self.train_df, self.test_df, rolling_window=window
            )

            # Save features for downstream
            self.train_df.to_csv(self.data_dir / "train_features.csv", index=False)
            self.test_df.to_csv(self.data_dir / "test_features.csv", index=False)

            models_dir = self.data_dir / "models"
            models_dir.mkdir(exist_ok=True)
            scaler_path = models_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            log.info(
                "Features and scaler saved", extra={"scaler_path": str(scaler_path)}
            )
        except Exception as e:
            raise DataError(f"Preprocessing failed: {e}") from e

    def run_training(self, model_name: Optional[str] = None) -> None:
        """Stage 3: Train model(s)."""
        try:
            log.info("Stage 3 — Training")
            if self.train_df is None:
                raise DataError("Run ingestion first")

            model_name = model_name or cfg.models.active_model
            feat_cols = get_feature_columns(self.train_df)

            healthy_limit = cfg.data.healthy_cycle_limit
            healthy_data = self.train_df[
                self.train_df["time_in_cycles"] <= healthy_limit
            ]

            model = self.registry.get_model(model_name)

            # Format inputs mathematically (sequence vs generic matrix)
            if model_name in ("lstm", "transformer"):
                X_train = create_sequences(
                    healthy_data, feat_cols, seq_len=cfg.models.lstm.sequence_length
                )
            else:
                X_train = healthy_data[feat_cols].to_numpy()

            t0 = time.perf_counter()
            model.train(X_train)  # type: ignore
            train_time = time.perf_counter() - t0
            log.info(
                f"Model '{model_name}' trained",
                extra={"samples": len(healthy_data), "time_sec": f"{train_time:.2f}"},
            )

            # Save model
            model_dir = self.data_dir / "models"
            model_dir.mkdir(exist_ok=True)
            model.save(model_dir / f"{model_name}.joblib")
        except Exception as e:
            raise ModelError(f"Training failed: {e}") from e

    def run_evaluation(self) -> dict[str, Any]:
        """Stage 4: Evaluate on test set."""
        try:
            log.info("Stage 4 — Evaluation")
            if self.test_df is None or self.rul is None:
                raise DataError("Run ingestion first")

            feat_cols = get_feature_columns(self.test_df)
            threshold = cfg.data.failure_rul_threshold

            y_true = create_labels(self.test_df, self.rul, failure_threshold=threshold)

            active_model = cfg.models.active_model
            if active_model in ("lstm", "transformer"):
                # Warning: Evaluation mapping sequence slices to labels requires alignment.
                # Since we predict sequence by sequence, we just chop off the first (seq_len - 1) labels per unit
                # or we just evaluate normally using sequences and shifted labels.
                X_test = create_sequences(
                    self.test_df, feat_cols, seq_len=cfg.models.lstm.sequence_length
                )

                # Align y_true for sequences: chop off first (seq_len - 1) cycles from y_true for each unit
                y_seqs = []
                seq_len = cfg.models.lstm.sequence_length
                idx = 0
                for unit in self.test_df["unit_number"].unique():
                    unit_len = len(self.test_df[self.test_df["unit_number"] == unit])
                    if unit_len >= seq_len:
                        y_seqs.extend(y_true[idx + seq_len - 1 : idx + unit_len])
                    idx += unit_len
                y_true = np.array(y_seqs)
            else:
                X_test = self.test_df[feat_cols].to_numpy()

            results: dict[str, Any] = self.registry.compare_all(X_test, y_true)  # type: ignore
            log.info("Evaluation complete", extra={"models": len(results)})
            return results
        except Exception as e:
            raise ModelError(f"Evaluation failed: {e}") from e

    def run_inference(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Stage 5: Run inference on test set, log to blockchain."""
        try:
            log.info("Stage 5 — Inference + Logging")
            if self.test_df is None:
                raise DataError("Run ingestion first")

            model_name = model_name or cfg.models.active_model
            model = self.registry.get_model(model_name)
            feat_cols = get_feature_columns(self.test_df)

            if model_name in ("lstm", "transformer"):
                X_infer = create_sequences(
                    self.test_df, feat_cols, seq_len=cfg.models.lstm.sequence_length
                )
                raw_predictions = model.predict(X_infer)

                # Re-align predictions back to dataframe (pad first seq_len-1 with 0 or NaN, or align indices)
                # For simplicity in blockchain, we just pad the beginning
                seq_len = cfg.models.lstm.sequence_length
                padded_predictions = []
                idx = 0
                for unit in self.test_df["unit_number"].unique():
                    unit_len = len(self.test_df[self.test_df["unit_number"] == unit])
                    if unit_len >= seq_len:
                        padded_predictions.extend([0] * (seq_len - 1))
                        padded_predictions.extend(
                            raw_predictions[idx : idx + unit_len - seq_len + 1]
                        )
                        idx += unit_len - seq_len + 1
                    else:
                        padded_predictions.extend([0] * unit_len)
                predictions = np.array(padded_predictions)
            else:
                X_infer = self.test_df[feat_cols].to_numpy()
                predictions = model.predict(X_infer)

            self.test_df["prediction"] = predictions

            # Log to blockchain (sample every 50th row)
            for idx in range(0, len(self.test_df), 50):
                row = self.test_df.iloc[idx]
                pred_val = int(predictions[idx])
                if pd.isna(pred_val):
                    continue

                self.ledger.add_entry(
                    unit_id=int(cast(float, row["unit_number"])),
                    cycles=int(cast(float, row["time_in_cycles"])),
                    health_score=float(1 - pred_val) * 100,
                    status="CRITICAL" if pred_val == 1 else "OPTIMAL",
                )

            log.info(
                "Inference complete",
                extra={
                    "anomalies": int(np.nansum(predictions)),
                    "total": len(predictions),
                },
            )
            return self.test_df
        except Exception as e:
            raise ModelError(f"Inference failed: {e}") from e

    def run_full_pipeline(self) -> dict[str, Any]:
        """Execute all stages end-to-end."""
        try:
            log.info("=== PulseNet Full Pipeline Start ===")
            t0 = time.perf_counter()

            self.run_ingestion()
            self.run_preprocessing()
            self.run_training()
            results = self.run_evaluation()
            self.run_inference()

            duration = time.perf_counter() - t0
            log.info(
                "=== Pipeline Complete ===", extra={"duration_sec": f"{duration:.2f}"}
            )
            return results
        except PulseNetError as e:
            log.error(
                f"Pipeline failed: {e.message}", extra={"error_code": e.error_code}
            )
            return {}
        except Exception as e:
            log.critical(f"Unhandled pipeline failure: {e}")
            return {}
