# pyre-ignore-all-errors
"""
Benchmarking suite — measures inference latency, throughput, network resilience,
and encryption overhead. Generates graphs and tables.
"""

# pyre-ignore-all-errors

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np  # pyre-ignore
import pandas as pd  # pyre-ignore

from pulsenet.logger import get_logger  # pyre-ignore
from pulsenet.evaluation.metrics import (  # pyre-ignore
    calculate_detection_metrics,
    calculate_lead_time,
    map_ground_truth_labels,
)

log = get_logger(__name__)


class BenchmarkSuite:
    """Performance benchmarks for PulseNet pipeline."""

    def __init__(self, output_dir: str = "./outputs/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}

    # ------------------------------------------------------------------
    # Inference Latency
    # ------------------------------------------------------------------
    def benchmark_inference_latency(
        self,
        model,
        X: np.ndarray,
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict:
        """Measure per-sample inference latency."""
        log.info("Benchmarking inference latency")

        # Warmup
        for _ in range(warmup):
            model.predict(X[:1])

        latencies = []
        for i in range(iterations):
            idx = i % len(X)
            sample = X[idx : idx + 1]
            gc.disable()
            t0 = time.perf_counter_ns()
            model.predict(sample)
            t1 = time.perf_counter_ns()
            gc.enable()
            latencies.append((t1 - t0) / 1e6)  # ms

        result = {
            "mean_ms": round(float(np.mean(latencies)), 3),
            "median_ms": round(float(np.median(latencies)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies, 99)), 3),
            "min_ms": round(float(np.min(latencies)), 3),
            "max_ms": round(float(np.max(latencies)), 3),
            "target_met": bool(np.median(latencies) < 50),
        }
        self.results["inference_latency"] = result
        log.info("Inference latency", extra=result)
        return result

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------
    def benchmark_throughput(
        self,
        model,
        X: np.ndarray,
        batch_sizes: list[int] | None = None,
    ) -> dict:
        """Measure throughput in samples/sec for different batch sizes."""
        batch_sizes = batch_sizes or [1, 8, 32, 64, 128, 256]
        log.info("Benchmarking throughput")

        results = {}
        for bs in batch_sizes:
            if bs > len(X):
                continue
            batch = X[:bs]
            # Warmup
            model.predict(batch)

            t0 = time.perf_counter()
            for _ in range(10):
                model.predict(batch)
            elapsed = time.perf_counter() - t0

            throughput = (bs * 10) / elapsed
            results[f"batch_{bs}"] = round(float(throughput))  # pyre-ignore

        self.results["throughput"] = results
        log.info("Throughput results", extra=results)
        return results

    # ------------------------------------------------------------------
    # Network Resilience
    # ------------------------------------------------------------------
    def benchmark_network_resilience(
        self,
        data: np.ndarray,
        loss_rates: list[float] | None = None,
        trials: int = 1000,
    ) -> dict:
        """Simulate packet loss and measure data integrity."""
        loss_rates = loss_rates or [0.10, 0.20, 0.30]
        log.info("Benchmarking network resilience")

        results = {}
        for rate in loss_rates:
            surviving = []
            for _ in range(trials):
                mask = np.random.random(len(data)) > rate
                surviving.append(mask.sum() / len(data))

            integrity = float(np.mean(surviving) * 100)
            results[f"loss_{int(rate * 100)}pct"] = {
                "data_integrity_pct": round(integrity, 2),
                "avg_surviving": round(float(np.mean(surviving) * len(data)), 1),
                "target_met": integrity > 95,
            }

        self.results["network_resilience"] = results
        log.info("Network resilience", extra=results)
        return results

    # ------------------------------------------------------------------
    # Encryption Overhead
    # ------------------------------------------------------------------
    def benchmark_encryption(
        self,
        encryption_mgr,
        sample_data: str = "sensor_value_12345.678",
        iterations: int = 1000,
    ) -> dict:
        """Measure encryption/decryption overhead."""
        log.info("Benchmarking encryption overhead")

        # Encrypt
        enc_latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            ct = encryption_mgr.encrypt(sample_data)
            enc_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        # Decrypt
        dec_latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            encryption_mgr.decrypt(ct)
            dec_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        result = {
            "encrypt_mean_ms": round(float(np.mean(enc_latencies)), 4),
            "encrypt_p95_ms": round(float(np.percentile(enc_latencies, 95)), 4),
            "decrypt_mean_ms": round(float(np.mean(dec_latencies)), 4),
            "decrypt_p95_ms": round(float(np.percentile(dec_latencies, 95)), 4),
        }
        self.results["encryption"] = result
        log.info("Encryption overhead", extra=result)
        return result

    # ------------------------------------------------------------------
    # Detection Quality & Lead Time
    # ------------------------------------------------------------------
    def benchmark_detection_quality(
        self,
        model,
        X_test: np.ndarray,
        df_test: pd.DataFrame,
        rul_truth: pd.Series,
        threshold_cycles: int = 30,
    ) -> dict:
        """Measure Precision, Recall, F1, and AUC."""
        log.info("Benchmarking detection quality")

        # Prepare labels
        y_true = map_ground_truth_labels(df_test, rul_truth, threshold_cycles)

        # Get scores (Anomaly scores)
        # Isolation Forest: lower is more anomalous. We invert it.
        raw_scores = model.decision_function(X_test)
        y_scores = -raw_scores

        # Use model's internal threshold if available, otherwise default to 0.0
        # (Since 0.0 is the split for sklearn's decision_function)
        best_threshold = getattr(model, "threshold", 0.0)
        if best_threshold is None:
            best_threshold = 0.0

        metrics = calculate_detection_metrics(y_true, y_scores, threshold=best_threshold)

        
        self.results["detection_quality"] = metrics
        log.info("Detection quality metrics", extra=metrics)
        return metrics

    def benchmark_lead_time(
        self,
        model,
        X_test: np.ndarray,
        df_test: pd.DataFrame,
        rul_truth: pd.Series,
        failure_threshold_cycles: int = 30,
    ) -> dict:
        """Measure average lead time from first failure prediction to actual failure."""
        log.info("Benchmarking lead time")

        # Get predictions
        # For simplicity, we use the default threshold (0) for -scores
        raw_scores = model.decision_function(X_test)
        y_pred = (raw_scores < 0).astype(int)

        metrics = calculate_lead_time(df_test, y_pred, rul_truth, failure_threshold_cycles)
        
        self.results["lead_time"] = metrics
        log.info("Lead time metrics", extra=metrics)
        return metrics

    # ------------------------------------------------------------------
    # Robustness (Noise & Missing Data)
    # ------------------------------------------------------------------
    def benchmark_robustness(
        self,
        model,
        X_test: np.ndarray,
        df_test: pd.DataFrame,
        rul_truth: pd.Series,
        noise_levels: list[float] | None = None,
        dropout_rates: list[float] | None = None,
        threshold_cycles: int = 30,
    ) -> dict:
        """Measure performance degradation under synthetic noise and missing data."""
        noise_levels = noise_levels or [0.01, 0.05, 0.10]
        dropout_rates = dropout_rates or [0.05, 0.10, 0.20]
        log.info("Benchmarking robustness")
        y_true = map_ground_truth_labels(df_test, rul_truth, threshold_cycles)
        
        # Baseline F1
        raw_scores = model.decision_function(X_test)
        best_threshold = getattr(model, "threshold", 0.0)
        if best_threshold is None:
            best_threshold = 0.0
            
        baseline_f1 = calculate_detection_metrics(y_true, -raw_scores, threshold=best_threshold)["f1"]

        results = {"baseline_f1": baseline_f1, "noise": {}, "dropout": {}}

        # Noise Robustness
        for sigma in noise_levels:
            noise = np.random.normal(0, sigma, X_test.shape)
            X_noisy = X_test + noise
            scores = -model.decision_function(X_noisy)
            f1 = calculate_detection_metrics(y_true, scores, threshold=best_threshold)["f1"]
            results["noise"][f"sigma_{sigma}"] = {
                "f1": f1,
                "degradation_pct": round((baseline_f1 - f1) / baseline_f1 * 100, 2) if baseline_f1 > 0 else 0
            }

        # Dropout Robustness (Missing Data)
        for rate in dropout_rates:
            X_dropped = X_test.copy()
            mask = np.random.random(X_test.shape) < rate
            X_dropped[mask] = 0  # Impute with 0 (simple baseline)
            scores = -model.decision_function(X_dropped)
            f1 = calculate_detection_metrics(y_true, scores, threshold=best_threshold)["f1"]
            results["dropout"][f"rate_{rate}"] = {
                "f1": f1,
                "degradation_pct": round((baseline_f1 - f1) / baseline_f1 * 100, 2) if baseline_f1 > 0 else 0
            }

        self.results["robustness"] = results
        log.info("Robustness results", extra=results)
        return results

    # ------------------------------------------------------------------
    # CPU / Memory profiling
    # ------------------------------------------------------------------
    def profile_resources(self) -> dict:
        """Capture current resource usage including GPU stats (via pynvml)."""
        import psutil  # pyre-ignore

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        result = {
            "cpu_percent": proc.cpu_percent(interval=1),
            "memory_rss_mb": round(mem.rss / 1024 / 1024, 1),
            "memory_vms_mb": round(mem.vms / 1024 / 1024, 1),
            "threads": proc.num_threads(),
        }

        # GPU Profiling via pynvml
        try:
            import pynvml  # pyre-ignore

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                gpus.append(
                    {
                        "gpu_id": i,
                        "utilization_pct": util.gpu,
                        "vram_used_mb": round(mem_info.used / 1024**2, 1),  # type: ignore
                        "vram_total_mb": round(mem_info.total / 1024**2, 1),  # type: ignore
                        "power_watts": round(power_w, 1),
                    }
                )
            if gpus:
                result["gpus"] = gpus
            pynvml.nvmlShutdown()
        except ImportError:
            pass  # No pynvml installed
        except Exception as e:
            log.warning(f"Could not profile GPUs: {e}")

        self.results["resources"] = result
        log.info("Resource profile", extra=result)
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_results(self) -> Path:
        """Save all benchmark results to JSON."""
        out = self.output_dir / "benchmark_results.json"
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        log.info(f"Benchmarks saved to {out}")
        return out

    def generate_report_table(self) -> str:
        """Generate markdown table of results."""
        lines = ["# PulseNet Benchmark Results\n"]

        if "inference_latency" in self.results:
            r = self.results["inference_latency"]
            lines.append("## Inference Latency\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in r.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        if "throughput" in self.results:
            r = self.results["throughput"]
            lines.append("## Throughput (samples/sec)\n")
            lines.append("| Batch Size | Throughput |")
            lines.append("|-----------|-----------|")
            for k, v in r.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        if "network_resilience" in self.results:
            r = self.results["network_resilience"]
            lines.append("## Network Resilience\n")
            lines.append("| Packet Loss | Data Integrity | Target Met |")
            lines.append("|------------|---------------|------------|")
            for k, v in r.items():
                lines.append(
                    f"| {k} | {v['data_integrity_pct']}% | {'[PASS]' if v['target_met'] else '[FAIL]'} |"
                )
            lines.append("")

        if "encryption" in self.results:
            r = self.results["encryption"]
            lines.append("## Encryption Overhead\n")
            lines.append("| Operation | Mean (ms) | P95 (ms) |")
            lines.append("|-----------|----------|---------|")
            lines.append(
                f"| Encrypt | {r['encrypt_mean_ms']} | {r['encrypt_p95_ms']} |"
            )
            lines.append(
                f"| Decrypt | {r['decrypt_mean_ms']} | {r['decrypt_p95_ms']} |"
            )
            lines.append("")

        if "detection_quality" in self.results:
            r = self.results["detection_quality"]
            lines.append("## Detection Quality\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in r.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        if "lead_time" in self.results:
            r = self.results["lead_time"]
            lines.append("## Lead Time\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in r.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        if "robustness" in self.results:
            r = self.results["robustness"]
            lines.append("## Robustness (F1 Score)\n")
            lines.append(f"**Baseline F1:** {r['baseline_f1']}\n")
            
            lines.append("### Noise Sensitivity")
            lines.append("| Noise (Sigma) | F1 | Degradation |")

    def generate_plots(self) -> None:
        """Generate benchmark visualization plots."""
        try:
            import matplotlib  # pyre-ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # pyre-ignore

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            # Plot 1: Throughput by batch size
            if "throughput" in self.results:
                data = self.results["throughput"]
                sizes = [int(k.split("_")[1]) for k in data.keys()]
                vals = list(data.values())
                axes[0].bar(range(len(sizes)), vals, color="#2ecc71", alpha=0.8)
                axes[0].set_xticks(range(len(sizes)))
                axes[0].set_xticklabels(sizes)
                axes[0].set_xlabel("Batch Size")
                axes[0].set_ylabel("Samples/sec")
                axes[0].set_title("Throughput by Batch Size")

            # Plot 2: Network resilience
            if "network_resilience" in self.results:
                data = self.results["network_resilience"]
                rates = [
                    k.replace("loss_", "").replace("pct", "%") for k in data.keys()
                ]
                integ = [v["data_integrity_pct"] for v in data.values()]
                axes[1].bar(range(len(rates)), integ, color="#3498db", alpha=0.8)
                axes[1].set_xticks(range(len(rates)))
                axes[1].set_xticklabels(rates)
                axes[1].set_xlabel("Packet Loss Rate")
                axes[1].set_ylabel("Data Integrity %")
                axes[1].set_title("Network Resilience")
                axes[1].axhline(y=95, color="red", linestyle="--", label="95% target")
                axes[1].legend()

            # Plot 3: Encryption overhead
            if "encryption" in self.results:
                data = self.results["encryption"]
                ops = ["Encrypt", "Decrypt"]
                means = [data["encrypt_mean_ms"], data["decrypt_mean_ms"]]
                p95s = [data["encrypt_p95_ms"], data["decrypt_p95_ms"]]
                x = range(len(ops))
                axes[2].bar(x, means, color="#9b59b6", alpha=0.8, label="Mean")
                axes[2].bar(x, p95s, color="#e74c3c", alpha=0.4, label="P95")
                axes[2].set_xticks(x)
                axes[2].set_xticklabels(ops)
                axes[2].set_ylabel("Latency (ms)")
                axes[2].set_title("Encryption Overhead")
                axes[2].legend()

            # Plot 4: Detection Quality (F1/AUC)
            if "detection_quality" in self.results:
                data = self.results["detection_quality"]
                metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
                vals = [data[m] for m in metrics]
                axes[3].bar(metrics, vals, color="#e67e22", alpha=0.8)
                axes[3].set_ylim(0, 1.05)
                axes[3].set_ylabel("Score")
                axes[3].set_title("Detection Quality Metrics")

            # Plot 5: Robustness (Noise Sensitivity)
            if "robustness" in self.results:
                data = self.results["robustness"]
                sigmas = [float(k.split("_")[1]) for k in data["noise"].keys()]
                f1s = [v["f1"] for v in data["noise"].values()]
                axes[4].plot([0] + sigmas, [data["baseline_f1"]] + f1s, marker='o', color="#e74c3c", label="Noise (sigma)")
                
                rates = [float(k.split("_")[1]) for k in data["dropout"].keys()]
                dropout_f1s = [v["f1"] for v in data["dropout"].values()]
                axes[4].plot([0] + rates, [data["baseline_f1"]] + dropout_f1s, marker='s', color="#34495e", label="Dropout (rate)")
                
                axes[4].set_ylim(0, 1.05)
                axes[4].set_xlabel("Degradation Level")
                axes[4].set_ylabel("F1 Score")
                axes[4].set_title("Robustness Analysis")
                axes[4].legend()

            # Plot 6: Lead Time Distribution (Mock summary or count)
            if "lead_time" in self.results:
                data = self.results["lead_time"]
                axes[5].text(0.5, 0.5, f"Avg Lead Time:\n{data['avg_lead_time']} cycles\n\nDetection Rate:\n{data['detection_rate']*100:.1f}%", 
                            ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
                axes[5].set_title("Lead Time Summary")
                axes[5].axis('off')

            plt.tight_layout()
            plt.savefig(self.output_dir / "benchmark_plots.png", dpi=150)
            plt.close()
            log.info("Benchmark plots saved")
        except ImportError:
            log.warning("matplotlib not available — skipping plots")
