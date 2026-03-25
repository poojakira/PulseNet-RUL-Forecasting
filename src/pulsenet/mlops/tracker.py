"""
MLOps — MLflow tracking, data drift detection, and auto-retrain triggers.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pulsenet.logger import get_logger

log = get_logger(__name__)


class MLOpsTracker:
    """Experiment tracking, drift detection, and model monitoring."""

    def __init__(
        self,
        experiment_name: str = "PulseNet_PredictiveMaintenance",
        tracking_uri: str = "mlruns",
        drift_threshold: float = 0.1,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.drift_threshold = drift_threshold
        self._mlflow_available = False
        self._reference_stats: Optional[dict] = None

        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow_available = True
            log.info("MLflow initialized", extra={"uri": tracking_uri, "experiment": experiment_name})
        except ImportError:
            log.warning("MLflow not installed — using local file tracking")

    # ------------------------------------------------------------------
    # Experiment Tracking
    # ------------------------------------------------------------------
    def log_training_run(
        self,
        params: dict,
        metrics: dict,
        model_path: Optional[str] = None,
        artifacts: Optional[list[str]] = None,
    ) -> str:
        """Log a training run to MLflow or local file."""
        if self._mlflow_available:
            return self._log_mlflow(params, metrics, model_path, artifacts)
        return self._log_local(params, metrics, model_path)

    def _log_mlflow(self, params, metrics, model_path, artifacts) -> str:
        import mlflow
        with mlflow.start_run() as run:
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            if model_path and Path(model_path).exists():
                mlflow.log_artifact(model_path)
            if artifacts:
                for a in artifacts:
                    if Path(a).exists():
                        mlflow.log_artifact(a)
            run_id = run.info.run_id
        log.info("MLflow run logged", extra={"run_id": run_id})
        return run_id

    def _log_local(self, params, metrics, model_path) -> str:
        """Fallback: log to local JSON file."""
        entry = {
            "timestamp": time.time(),
            "params": params,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "model_path": model_path,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()

        log_file = Path(self.tracking_uri) / "local_tracking.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        log.info("Local tracking logged", extra={"hash": entry["hash"][:16]})
        return entry["hash"]

    # ------------------------------------------------------------------
    # Data Drift Detection
    # ------------------------------------------------------------------
    def set_reference_distribution(self, X: np.ndarray) -> None:
        """Store reference feature statistics for drift detection."""
        self._reference_stats = {
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0).tolist(),
            "shape": list(X.shape),
        }
        log.info("Reference distribution set", extra={"features": X.shape[1]})

    def detect_drift(self, X_new: np.ndarray) -> dict:
        """Detect data drift using KL divergence approximation.

        Compares incoming data distribution to reference.
        Returns drift metrics and whether retrain is recommended.
        """
        if self._reference_stats is None:
            return {"drift_detected": False, "message": "No reference distribution set"}

        ref_mean = np.array(self._reference_stats["mean"])
        ref_std = np.array(self._reference_stats["std"])
        new_mean = X_new.mean(axis=0)
        new_std = X_new.std(axis=0)

        # Approximate KL divergence per feature (normal assumption)
        epsilon = 1e-8
        kl_per_feature = (
            np.log((new_std + epsilon) / (ref_std + epsilon))
            + (ref_std ** 2 + (ref_mean - new_mean) ** 2) / (2 * (new_std + epsilon) ** 2)
            - 0.5
        )
        avg_kl = float(np.mean(np.abs(kl_per_feature)))

        drift_detected = avg_kl > self.drift_threshold

        result = {
            "avg_kl_divergence": round(avg_kl, 6),
            "drift_threshold": self.drift_threshold,
            "drift_detected": drift_detected,
            "retrain_recommended": drift_detected,
            "drifted_features": int(np.sum(np.abs(kl_per_feature) > self.drift_threshold)),
        }

        if drift_detected:
            log.warning("Data drift detected!", extra=result)
        else:
            log.info("No significant drift", extra=result)

        return result

    # ------------------------------------------------------------------
    # Model Performance Monitoring
    # ------------------------------------------------------------------
    def log_inference_metrics(
        self,
        predictions: np.ndarray,
        latency_ms: float,
        batch_size: int,
    ) -> None:
        """Track inference-time metrics for monitoring."""
        anomaly_rate = float(predictions.sum() / len(predictions))
        entry = {
            "timestamp": time.time(),
            "anomaly_rate": round(anomaly_rate, 4),
            "latency_ms": round(latency_ms, 2),
            "batch_size": batch_size,
        }
        if self._mlflow_available:
            try:
                import mlflow
                for k, v in entry.items():
                    if k != "timestamp":
                        mlflow.log_metric(f"inference_{k}", float(v))
            except Exception:
                pass
        log.debug("Inference metrics logged", extra=entry)
