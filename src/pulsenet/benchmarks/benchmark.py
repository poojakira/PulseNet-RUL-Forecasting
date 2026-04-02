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

from pulsenet.logger import get_logger  # pyre-ignore

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
                    f"| {k} | {v['data_integrity_pct']}% | {'✅' if v['target_met'] else '❌'} |"
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

        report = "\n".join(lines)
        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        return report

    def generate_plots(self) -> None:
        """Generate benchmark visualization plots."""
        try:
            import matplotlib  # pyre-ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # pyre-ignore

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

            plt.tight_layout()
            plt.savefig(self.output_dir / "benchmark_plots.png", dpi=150)
            plt.close()
            log.info("Benchmark plots saved")
        except ImportError:
            log.warning("matplotlib not available — skipping plots")
