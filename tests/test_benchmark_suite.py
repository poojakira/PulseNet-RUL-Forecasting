"""Tests for the benchmarking suite and evaluation metrics."""

from __future__ import annotations

import numpy as np

from pulsenet.benchmarks.benchmark import BenchmarkSuite
from pulsenet.evaluation.metrics import (
    calculate_detection_metrics,
    calculate_lead_time,
    map_ground_truth_labels,
)
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.security.encryption import EncryptionManager


def _model_and_data(official_fd001):
    df = official_fd001.test.copy()
    feat_cols = [c for c in df.columns if str(c).startswith("sensor_")]
    x = df[feat_cols].to_numpy()
    model = IsolationForestModel(n_estimators=30, contamination=0.1)
    model.train(x)
    return model, x, df, official_fd001.rul


class TestBenchmarkSuite:
    def test_full_benchmark_flow(self, official_fd001, tmp_path):
        model, x, df, rul = _model_and_data(official_fd001)
        suite = BenchmarkSuite(output_dir=str(tmp_path / "bench"))

        lat = suite.benchmark_inference_latency(model, x, warmup=2, iterations=5)
        assert "mean_ms" in lat

        tp = suite.benchmark_throughput(model, x, batch_sizes=[1, 8])
        assert tp

        net = suite.benchmark_network_resilience(x, loss_rates=[0.1], trials=20)
        assert "loss_10pct" in net

        enc = EncryptionManager(key_file=str(tmp_path / "k.key"), rotation_days=30)
        enc_res = suite.benchmark_encryption(enc, iterations=20)
        assert "encrypt_mean_ms" in enc_res

        dq = suite.benchmark_detection_quality(model, x, df, rul, threshold_cycles=30)
        assert "f1" in dq

        lt = suite.benchmark_lead_time(model, x, df, rul)
        assert "avg_lead_time" in lt

        rob = suite.benchmark_robustness(
            model, x, df, rul, noise_levels=[0.01], dropout_rates=[0.05]
        )
        assert "baseline_f1" in rob

        res = suite.profile_resources()
        assert "memory_rss_mb" in res

        out = suite.save_results()
        assert out.exists()

        report = suite.generate_report_table()
        assert "PulseNet Benchmark Results" in report

        suite.generate_plots()


class TestEvaluationMetrics:
    def test_detection_metrics_two_class(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        m = calculate_detection_metrics(y_true, y_scores, threshold=0.5)
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_detection_metrics_single_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        m = calculate_detection_metrics(y_true, y_scores)
        assert m["roc_auc"] == 0.0
        assert m["pr_auc"] == 0.0

    def test_map_labels_and_lead_time(self, official_fd001):
        df = official_fd001.test.copy()
        rul = official_fd001.rul
        y = map_ground_truth_labels(df, rul, threshold_cycles=30)
        assert len(y) == len(df)

        y_pred = np.ones(len(df), dtype=int)
        lt = calculate_lead_time(df, y_pred, rul, failure_threshold_cycles=30)
        assert lt["engines_detected"] >= 0

    def test_lead_time_no_detections(self, official_fd001):
        df = official_fd001.test.copy()
        rul = official_fd001.rul
        lt = calculate_lead_time(df, np.zeros(len(df), dtype=int), rul)
        assert lt["engines_detected"] == 0
