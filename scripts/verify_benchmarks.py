#!/usr/bin/env python3
"""
Benchmark verification script — runs the full benchmark suite and saves results.

Usage:
    python scripts/verify_benchmarks.py

Requires:
    - Trained model in models/isolation_forest.joblib
    - Preprocessed test features in data/test_features.csv
    - Ground truth RUL in data/RUL_FD001.txt
"""

import sys
from pathlib import Path

# Add src to sys.path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from pulsenet.benchmarks.benchmark import BenchmarkSuite
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.pipeline.preprocessing import get_feature_columns
from pulsenet.security.encryption import EncryptionManager

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent.parent / "models" / "isolation_forest.joblib"
TEST_FEATURES = DATA_DIR / "test_features.csv"
RUL_TRUTH_FILE = DATA_DIR / "RUL_FD001.txt"

# Standard C-MAPSS failure threshold (RUL <= 30 cycles = approaching failure)
FAILURE_THRESHOLD_CYCLES = 30


def run_verification():
    """Run full benchmark suite and save results."""
    print("--- STARTING BENCHMARK VERIFICATION ---")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run 'python main_pipeline.py --mode train' first.")
        return

    if not TEST_FEATURES.exists():
        print(f"ERROR: Test features not found at {TEST_FEATURES}")
        print("Run 'python main_pipeline.py --mode full' first.")
        return

    # Load resources
    print("Loading model and data...")
    model = IsolationForestModel()
    model.load(MODEL_PATH)

    df_test = pd.read_csv(TEST_FEATURES)
    rul_truth = pd.read_csv(RUL_TRUTH_FILE, header=None).iloc[:, 0]

    feat_cols = get_feature_columns(df_test)
    X_test = df_test[feat_cols].values

    print(f"Data shape: {X_test.shape}")
    print(f"Features: {len(feat_cols)}")
    print(f"Failure threshold: RUL <= {FAILURE_THRESHOLD_CYCLES} cycles")

    # Initialize Benchmark Suite
    suite = BenchmarkSuite(output_dir="./outputs/benchmarks")

    # 1. Performance benchmarks
    print("\n[1/6] Inference Latency...")
    suite.benchmark_inference_latency(model, X_test[:100])

    print("[2/6] Throughput...")
    suite.benchmark_throughput(model, X_test[:1000])

    # 2. Network & Security
    print("[3/6] Network Resilience...")
    suite.benchmark_network_resilience(X_test[:500])

    print("[4/6] Encryption Overhead...")
    suite.benchmark_encryption(EncryptionManager())

    # 3. Detection Quality
    print("[5/6] Detection Quality...")
    suite.benchmark_detection_quality(
        model, X_test, df_test, rul_truth, threshold_cycles=FAILURE_THRESHOLD_CYCLES
    )

    print("[6/6] Lead Time & Robustness...")
    suite.benchmark_lead_time(
        model, X_test, df_test, rul_truth,
        failure_threshold_cycles=FAILURE_THRESHOLD_CYCLES
    )
    suite.benchmark_robustness(
        model, X_test, df_test, rul_truth,
        threshold_cycles=FAILURE_THRESHOLD_CYCLES
    )

    # Export
    print("\nSaving results...")
    suite.save_results()
    report = suite.generate_report_table()
    print(report)

    try:
        suite.generate_plots()
        print(f"Plots saved to {suite.output_dir}/benchmark_plots.png")
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")

    print(f"\nDone. Results: {suite.output_dir}/benchmark_results.json")


if __name__ == "__main__":
    run_verification()
