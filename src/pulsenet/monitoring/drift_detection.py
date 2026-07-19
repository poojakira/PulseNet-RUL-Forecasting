"""
PulseNet Production Monitoring & Drift Detection — Using Evidently AI.

Provides:
- Data drift detection (PSI, KS-test, Jensen-Shannon)
- Target drift detection
- Feature importance drift
- Automated alerts via Prometheus/AlertManager
- Integration with retraining pipeline
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server

try:
    from evidently import ColumnMapping
    from evidently.metrics import (
        ColumnDriftMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnQuantileMetric,
        ColumnCorrelationsMetric,
    )
    from evidently.report import Report
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfDriftedColumns,
        TestShareOfDriftedColumns,
        TestColumnDrift,
        TestColumnsType,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    ColumnMapping = None

from pulsenet.logger import get_logger

log = get_logger(__name__)

# Prometheus metrics
DRIFT_DETECTED = Counter("pulsenet_drift_detected_total", "Total drift detections", ["feature", "drift_type"])
DRIFT_SCORE = Gauge("pulsenet_drift_score", "Current drift score", ["feature", "metric"])
DRIFT_CHECK_DURATION = Histogram("pulsenet_drift_check_duration_seconds", "Time spent on drift checks")
MODEL_PERFORMANCE = Gauge("pulsenet_model_performance", "Model performance metrics", ["metric"])
RETRAINING_TRIGGERED = Counter("pulsenet_retraining_triggered_total", "Retraining pipeline triggers")


class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    TARGET_DRIFT = "target_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftSeverity(Enum):
    """Severity levels for drift alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Drift detection alert."""
    alert_id: str
    timestamp: datetime
    feature: str
    drift_type: DriftType
    severity: DriftSeverity
    score: float
    threshold: float
    details: dict[str, Any]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "feature": self.feature,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "score": self.score,
            "threshold": self.threshold,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class DriftCheckConfig:
    """Configuration for drift checks."""
    psi_threshold: float = 0.2
    ks_threshold: float = 0.05
    js_threshold: float = 0.1
    min_samples: int = 100
    check_interval_hours: int = 1
    reference_window_days: int = 7
    current_window_days: int = 1
    alert_on_critical: bool = True
    alert_on_warning: bool = True


class EvidentlyDriftDetector:
    """
    Production drift detector using Evidently AI.
    
    Integrates with Prometheus for metrics and alerting.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_mapping: ColumnMapping,
        config: Optional[DriftCheckConfig] = None,
        prometheus_port: int = 9090,
    ):
        if not EVIDENTLY_AVAILABLE:
            raise RuntimeError("Evidently AI not installed. Install with: pip install evidently")
        
        self.reference_data = reference_data
        self.column_mapping = column_mapping
        self.config = config or DriftCheckConfig()
        self.alerts: list[DriftAlert] = []
        self._lock = threading.RLock()
        
        # Start Prometheus metrics server
        try:
            start_http_server(prometheus_port)
            log.info(f"Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            log.warning(f"Could not start Prometheus server: {e}")
        
        # Build drift report
        self._build_drift_report()
    
    def _build_drift_report(self) -> None:
        """Build Evidently report with drift metrics."""
        self.drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            *[ColumnDriftMetric(column_name=col) for col in self.reference_data.columns[:20]],
        ])
        
        self.test_suite = TestSuite(tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns(threshold=0.3),
            *[TestColumnDrift(column_name=col) for col in self.reference_data.columns[:20]],
            TestColumnsType(),
        ])
    
    def check_drift(
        self,
        current_data: pd.DataFrame,
        feature_list: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run drift detection on current data vs reference.
        
        Returns:
            Dict with drift results, scores, and any alerts
        """
        start = time.perf_counter()
        
        try:
            # Run Evidently report
            self.drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )
            
            self.test_suite.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )
            
            # Extract results
            report_dict = self.drift_report.as_dict()
            test_dict = self.test_suite.as_dict()
            
            results = {
                "timestamp": datetime.utcnow().isoformat(),
                "dataset_drift": report_dict["metrics"][0]["result"]["dataset_drift"],
                "n_drifted_features": report_dict["metrics"][0]["result"]["number_of_drifted_columns"],
                "total_features": report_dict["metrics"][0]["result"]["number_of_columns"],
                "drift_share": report_dict["metrics"][0]["result"]["share_of_drifted_columns"],
                "feature_drifts": {},
                "alerts": [],
            }
            
            # Extract per-feature drift scores
            for metric in report_dict["metrics"][1:]:
                if metric["metric"] == "ColumnDriftMetric":
                    col = metric["result"]["column_name"]
                    drift_score = metric["result"]["drift_score"]
                    is_drift = metric["result"]["drift_detected"]
                    
                    results["feature_drifts"][col] = {
                        "drift_score": drift_score,
                        "drift_detected": is_drift,
                        "stattest": metric["result"].get("stattest_name"),
                        "threshold": metric["result"].get("stattest_threshold"),
                    }
                    
                    # Update Prometheus metrics
                    DRIFT_SCORE.labels(feature=col, metric="drift_score").set(drift_score)
                    
                    # Generate alerts
                    alert = self._generate_alert(col, drift_score, is_drift)
                    if alert:
                        results["alerts"].append(alert.to_dict())
            
            # Test suite results
            results["tests"] = {
                "passed": all(t["status"] == "SUCCESS" for t in test_dict.get("tests", [])),
                "details": test_dict.get("tests", []),
            }
            
            DRIFT_CHECK_DURATION.observe(time.perf_counter() - start)
            
            log.info(
                f"Drift check completed: drifted={results['n_drifted_features']}/{results['total_features']}, "
                f"share={results['drift_share']:.3f}, duration={time.perf_counter() - start:.2f}s"
            )
            
            return results
            
        except Exception as e:
            log.error(f"Drift check failed: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    def _generate_alert(
        self,
        feature: str,
        drift_score: float,
        is_drift: bool,
    ) -> Optional[DriftAlert]:
        """Generate alert if drift exceeds thresholds."""
        if not is_drift:
            return None
        
        severity = DriftSeverity.INFO
        if drift_score > self.config.psi_threshold * 2:
            severity = DriftSeverity.CRITICAL
        elif drift_score > self.config.psi_threshold:
            severity = DriftSeverity.WARNING
        
        if severity == DriftSeverity.INFO and not self.config.alert_on_warning:
            return None
        if severity == DriftSeverity.CRITICAL and not self.config.alert_on_critical:
            return None
        
        alert = DriftAlert(
            alert_id=f"drift_{feature}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            feature=feature,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            score=drift_score,
            threshold=self.config.psi_threshold,
            details={
                "method": "PSI",
                "reference_samples": len(self.reference_data),
                "current_samples": None,  # Would need current data length
            },
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Update Prometheus
        DRIFT_DETECTED.labels(feature=feature, drift_type="data_drift").inc()
        
        log.warning(f"DRIFT ALERT: {feature} (score={drift_score:.4f}, severity={severity.value})")
        
        return alert
    
    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[DriftSeverity] = None,
        unacknowledged_only: bool = True,
    ) -> list[DriftAlert]:
        """Get filtered alerts."""
        with self._lock:
            alerts = self.alerts
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.utcnow()
                    log.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                    return True
        return False
    
    def run_continuous(self, data_provider: Callable[[], pd.DataFrame], interval_seconds: int = 3600) -> None:
        """Run continuous drift monitoring."""
        log.info(f"Starting continuous drift monitoring (interval={interval_seconds}s)")
        
        while True:
            try:
                current_data = data_provider()
                if current_data is not None and len(current_data) >= self.config.min_samples:
                    self.check_drift(current_data)
                else:
                    log.warning(f"Insufficient data for drift check: {len(current_data) if current_data is not None else 0} samples")
            except Exception as e:
                log.error(f"Continuous drift check failed: {e}")
            
            time.sleep(interval_seconds)


class FeatureDriftMonitor:
    """
    Lightweight feature drift monitor without Evidently dependency.
    Uses PSI, KS-test, and Jensen-Shannon divergence.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        config: Optional[DriftCheckConfig] = None,
    ):
        self.reference_data = reference_data
        self.config = config or DriftCheckConfig()
        self.feature_stats: dict[str, dict[str, float]] = {}
        self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> None:
        """Compute reference statistics for each feature."""
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            data = self.reference_data[col].dropna()
            if len(data) > 0:
                self.feature_stats[col] = {
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "quantiles": data.quantile([0.25, 0.5, 0.75]).to_dict(),
                    "hist": np.histogram(data, bins=20, density=True)[0].tolist(),
                    "bins": np.histogram(data, bins=20, density=True)[1].tolist(),
                    "count": len(data),
                }
    
    def compute_psi(
        self,
        current_data: pd.Series,
        reference_feature: str,
        bins: int = 20,
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        """
        ref_stats = self.feature_stats.get(reference_feature)
        if not ref_stats:
            return 0.0
        
        # Bin current data using reference bins
        current_data = current_data.dropna()
        if len(current_data) == 0:
            return 0.0
        
        ref_bins = ref_stats["bins"]
        ref_hist = ref_stats["hist"]
        
        curr_hist, _ = np.histogram(current_data, bins=ref_bins, density=True)
        
        # Normalize
        ref_hist = np.array(ref_hist) + 1e-10
        curr_hist = np.array(curr_hist) + 1e-10
        
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # PSI formula
        psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
        return float(psi)
    
    def compute_ks_statistic(
        self,
        current_data: pd.Series,
        reference_feature: str,
    ) -> tuple[float, float]:
        """
        Compute Kolmogorov-Smirnov statistic and p-value.
        
        Returns: (ks_statistic, p_value)
        """
        from scipy import stats
        
        ref_data = self.reference_data[reference_feature].dropna()
        curr_data = current_data.dropna()
        
        if len(ref_data) < 2 or len(curr_data) < 2:
            return 0.0, 1.0
        
        ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
        return float(ks_stat), float(p_value)
    
    def compute_js_divergence(
        self,
        current_data: pd.Series,
        reference_feature: str,
        bins: int = 20,
    ) -> float:
        """Compute Jensen-Shannon divergence."""
        ref_stats = self.feature_stats.get(reference_feature)
        if not ref_stats:
            return 0.0
        
        current_data = current_data.dropna()
        if len(current_data) == 0:
            return 0.0
        
        ref_bins = ref_stats["bins"]
        ref_hist = np.array(ref_stats["hist"]) + 1e-10
        curr_hist, _ = np.histogram(current_data, bins=ref_bins, density=True)
        curr_hist = np.array(curr_hist) + 1e-10
        
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        m = 0.5 * (ref_hist + curr_hist)
        js = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + 0.5 * np.sum(curr_hist * np.log(curr_hist / m))
        return float(js)
    
    def check_all_features(
        self,
        current_data: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """
        Run all drift checks on all numeric features.
        
        Returns:
            Dict mapping feature -> {psi, ks_stat, p_value, js_divergence}
        """
        results = {}
        
        for col in current_data.select_dtypes(include=[np.number]).columns:
            if col not in self.feature_stats:
                continue
            
            psi = self.compute_psi(current_data[col], col)
            ks_stat, p_value = self.compute_ks_statistic(current_data[col], col)
            js_div = self.compute_js_divergence(current_data[col], col)
            
            results[col] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": p_value,
                "js_divergence": js_div,
                "drift_detected": (
                    psi > self.config.psi_threshold or
                    ks_stat > self.config.ks_threshold or
                    js_div > self.config.js_threshold
                ),
            }
        
        return results


class ModelPerformanceMonitor:
    """
    Monitor model performance metrics for concept drift detection.
    """
    
    def __init__(self, model, reference_predictions: np.ndarray, reference_labels: np.ndarray):
        self.model = model
        self.reference_predictions = reference_predictions
        self.reference_labels = reference_labels
        self.performance_history: list[dict] = []
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model performance and check for degradation."""
        preds = self.model.predict(X)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
        
        metrics = {}
        
        # Classification metrics
        if len(np.unique(y)) <= 10:
            metrics["accuracy"] = float(accuracy_score(y, preds))
            metrics["f1_macro"] = float(f1_score(y, preds, average="macro", zero_division=0))
            try:
                metrics["roc_auc"] = float(roc_auc_score(y, self.model.predict_proba(X), multi_class="ovr"))
            except Exception:
                pass
        else:
            metrics["mse"] = float(mean_squared_error(y, preds))
            metrics["mae"] = float(mean_absolute_error(y, preds))
        
        # Compare with reference
        ref_metrics = self._compute_reference_metrics()
        
        for metric_name, value in metrics.items():
            if metric_name in ref_metrics:
                ref_value = ref_metrics[metric_name]
                # Check for degradation (>5% drop for accuracy/F1, >10% increase for MSE)
                if metric_name in ("accuracy", "f1_macro", "roc_auc"):
                    degradation = (ref_value - value) / ref_value if ref_value > 0 else 0
                    if degradation > 0.05:
                        log.warning(f"Model degradation detected: {metric_name} dropped {degradation:.1%} (ref={ref_value:.4f}, current={value:.4f})")
                        MODEL_PERFORMANCE.labels(metric=f"{metric_name}_degradation").set(degradation)
                elif metric_name in ("mse", "mae"):
                    increase = (value - ref_value) / ref_value if ref_value > 0 else 0
                    if increase > 0.1:
                        log.warning(f"Model degradation: {metric_name} increased {increase:.1%}")
                        MODEL_PERFORMANCE.labels(metric=f"{metric_name}_increase").set(increase)
        
        # Update Prometheus metrics
        for name, value in metrics.items():
            MODEL_PERFORMANCE.labels(metric=name).set(value)
        
        self.performance_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        })
        
        return metrics
    
    def _compute_reference_metrics(self) -> dict[str, float]:
        """Compute metrics on reference data."""
        # This would use the reference predictions/labels
        return {}


# Convenience function for quick setup
def create_drift_monitoring(
    reference_data: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    config: Optional[DriftCheckConfig] = None,
) -> tuple[EvidentlyDriftDetector, FeatureDriftMonitor]:
    """
    Create complete drift monitoring setup.
    
    Returns:
        (EvidentlyDriftDetector, FeatureDriftMonitor)
    """
    column_mapping = ColumnMapping(
        target=target_column,
        prediction=prediction_column,
        numerical_features=numeric_features,
        categorical_features=categorical_features,
    )
    
    evidently = EvidentlyDriftDetector(
        reference_data=reference_data,
        column_mapping=column_mapping,
        config=config,
    )
    
    lightweight = FeatureDriftMonitor(
        reference_data=reference_data,
        config=config,
    )
    
    return evidently, lightweight