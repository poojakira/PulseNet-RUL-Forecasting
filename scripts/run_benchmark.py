#!/usr/bin/env python3
"""
Production benchmark runner for PulseNet.

Measures and records:
  - Inference latency (mean, p50, p95, p99)
  - Throughput (samples/sec)
  - Encryption overhead (encrypt/decrypt latency)

Usage:
    python scripts/run_benchmark.py [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["PULSENET_ENV"] = "benchmark"


def _fmt(val: float, decimals: int = 3) -> float:
    return round(float(val), decimals)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def benchmark_inference_latency(
    model, X: np.ndarray, warmup: int = 50, iterations: int = 500
) -> dict:
    """Inference latency (ms) with GC-disabled hot loop."""
    import gc

    for _ in range(warmup):
        model.predict(X[:1])

    latencies: list[float] = []
    for i in range(iterations):
        sample = X[i % len(X) : i % len(X) + 1]
        gc.disable()
        t0 = time.perf_counter_ns()
        model.predict(sample)
        t1 = time.perf_counter_ns()
        gc.enable()
        latencies.append((t1 - t0) / 1e6)

    arr = np.array(latencies)
    return {
        "mean_ms": _fmt(np.mean(arr)),
        "p50_ms": _fmt(np.median(arr)),
        "p95_ms": _fmt(np.percentile(arr, 95)),
        "p99_ms": _fmt(np.percentile(arr, 99)),
        "min_ms": _fmt(np.min(arr)),
        "max_ms": _fmt(np.max(arr)),
        "samples": iterations,
    }


def benchmark_throughput(
    model, X: np.ndarray, batch_sizes: list[int] | None = None
) -> dict:
    """Throughput in samples/sec for various batch sizes."""
    batch_sizes = batch_sizes or [1, 8, 32, 64, 128, 256]
    results: dict[str, float] = {}
    for bs in batch_sizes:
        if bs > len(X):
            continue
        batch = X[:bs]
        model.predict(batch)
        t0 = time.perf_counter()
        for _ in range(20):
            model.predict(batch)
        elapsed = time.perf_counter() - t0
        samples_per_sec = (bs * 20) / elapsed
        results[f"batch_{bs}"] = int(samples_per_sec)
    return results


def benchmark_encryption_overhead(encryption_mgr, iterations: int = 2000) -> dict:
    """Encrypt/decrypt latency in ms."""
    sample = "sensor_value_12345.67890"

    enc_latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        encryption_mgr.encrypt(sample)
        enc_latencies.append((time.perf_counter_ns() - t0) / 1e6)

    ct = encryption_mgr.encrypt(sample)
    dec_latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        encryption_mgr.decrypt(ct)
        dec_latencies.append((time.perf_counter_ns() - t0) / 1e6)

    enc_arr = np.array(enc_latencies)
    dec_arr = np.array(dec_latencies)
    return {
        "encrypt_mean_ms": _fmt(np.mean(enc_arr), 4),
        "encrypt_p50_ms": _fmt(np.median(enc_arr), 4),
        "encrypt_p95_ms": _fmt(np.percentile(enc_arr, 95), 4),
        "encrypt_p99_ms": _fmt(np.percentile(enc_arr, 99), 4),
        "decrypt_mean_ms": _fmt(np.mean(dec_arr), 4),
        "decrypt_p50_ms": _fmt(np.median(dec_arr), 4),
        "decrypt_p95_ms": _fmt(np.percentile(dec_arr, 95), 4),
        "decrypt_p99_ms": _fmt(np.percentile(dec_arr, 99), 4),
        "samples": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PulseNet benchmark runner")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PULSENET BENCHMARK RUNNER")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading NASA C-MAPSS FD001 data...")
    from pulsenet.pipeline.official_cmapss import load_official_fd001
    from pulsenet.pipeline.preprocessing import (
        compute_rolling_features,
        get_feature_columns,
        normalize,
    )

    fd001 = load_official_fd001("data/official", max_train_rows=4000, download=False)
    train_df = compute_rolling_features(fd001.train.copy())
    test_df = compute_rolling_features(fd001.test.copy())
    train_df, test_df, _ = normalize(train_df, test_df)
    feature_cols = get_feature_columns(train_df)
    X_test = test_df[feature_cols].to_numpy()
    print(f"  Features: {len(feature_cols)}, Test samples: {len(X_test)}")

    # 2. Train model
    print("\n[2/5] Training Isolation Forest...")
    from pulsenet.models.isolation_forest import IsolationForestModel

    healthy_train = train_df[train_df["time_in_cycles"] <= 40][feature_cols].to_numpy()
    model = IsolationForestModel(n_estimators=100, contamination=0.12)
    t0 = time.perf_counter()
    model.train(healthy_train)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.3f}s")

    # 3. Inference latency & throughput
    print("\n[3/5] Benchmarking inference...")
    latency = benchmark_inference_latency(model, X_test)
    print(
        f"  Mean: {latency['mean_ms']}ms  p50: {latency['p50_ms']}ms  "
        f"p95: {latency['p95_ms']}ms  p99: {latency['p99_ms']}ms"
    )

    throughput = benchmark_throughput(model, X_test)
    print(f"  Throughput (batch_128): {throughput.get('batch_128', 'N/A')} samples/sec")

    # 4. Encryption overhead
    print("\n[4/5] Benchmarking encryption...")
    from pulsenet.security.encryption import EncryptionManager

    enc_mgr = EncryptionManager()
    encryption = benchmark_encryption_overhead(enc_mgr)
    print(
        f"  Encrypt mean: {encryption['encrypt_mean_ms']}ms  "
        f"Decrypt mean: {encryption['decrypt_mean_ms']}ms"
    )

    # 5. Resource snapshot
    print("\n[5/5] Capturing resource snapshot...")
    import psutil

    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    resources = {
        "cpu_percent": proc.cpu_percent(interval=1),
        "memory_rss_mb": _fmt(mem.rss / 1024 / 1024, 1),
        "memory_vms_mb": _fmt(mem.vms / 1024 / 1024, 1),
        "threads": proc.num_threads(),
    }
    print(f"  RSS: {resources['memory_rss_mb']}MB  CPU: {resources['cpu_percent']}%")

    # Compile results
    results = {
        "benchmark_version": "3.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        },
        "model": {
            "name": "isolation_forest",
            "n_estimators": 100,
            "train_time_seconds": _fmt(train_time, 3),
        },
        "inference_latency_ms": latency,
        "throughput_samples_per_sec": throughput,
        "encryption_overhead_ms": encryption,
        "resources": resources,
    }

    # Save JSON
    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {json_path}")

    # Save markdown report
    report = _generate_report(results)
    md_path = output_dir / "benchmark_report.md"
    md_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {md_path}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    return 0


def _generate_report(results: dict) -> str:
    lines = ["# PulseNet Benchmark Results\n"]
    lines.append(f"- **Version**: {results['benchmark_version']}")
    lines.append(f"- **Timestamp**: {results['timestamp']}")
    lines.append(f"- **Platform**: {results['system']['platform']}")
    lines.append(f"- **Python**: {results['system']['python_version']}")
    lines.append(f"- **Model**: {results['model']['name']}")
    lines.append("")

    # Inference latency
    lat = results["inference_latency_ms"]
    lines.append("## Inference Latency (ms)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in lat.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Throughput
    tp = results["throughput_samples_per_sec"]
    lines.append("## Throughput (samples/sec)\n")
    lines.append("| Batch Size | Throughput |")
    lines.append("|-----------|-----------|")
    for k, v in tp.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Encryption
    enc = results["encryption_overhead_ms"]
    lines.append("## Encryption Overhead (ms)\n")
    lines.append("| Operation | Mean | p50 | p95 | p99 |")
    lines.append("|-----------|------|-----|-----|-----|")
    lines.append(
        f"| Encrypt | {enc['encrypt_mean_ms']} | {enc['encrypt_p50_ms']} | "
        f"{enc['encrypt_p95_ms']} | {enc['encrypt_p99_ms']} |"
    )
    lines.append(
        f"| Decrypt | {enc['decrypt_mean_ms']} | {enc['decrypt_p50_ms']} | "
        f"{enc['decrypt_p95_ms']} | {enc['decrypt_p99_ms']} |"
    )
    lines.append("")

    # Resources
    res = results["resources"]
    lines.append("## Resource Snapshot\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in res.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
