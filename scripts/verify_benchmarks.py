import sys
import os
from pathlib import Path

# Add src to sys.path for local imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import joblib
from pulsenet.benchmarks.benchmark import BenchmarkSuite
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.pipeline.preprocessing import get_feature_columns
from pulsenet.security.encryption import EncryptionManager

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent.parent / "models" / "isolation_forest.joblib"
TEST_FEATURES = DATA_DIR / "test_features.csv"
RUL_TRUTH_FILE = DATA_DIR / "RUL_FD001.txt"

def run_verification():
    print("--- STARTING BENCHMARK VERIFICATION (V3) ---")
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # Load resources
    print("Loading model and data...")
    model_wrapper = IsolationForestModel()
    model_wrapper.load(MODEL_PATH)
    
    df_test = pd.read_csv(TEST_FEATURES)
    rul_truth = pd.read_csv(RUL_TRUTH_FILE, header=None).iloc[:, 0]
    
    # Feature names
    feat_cols = get_feature_columns(df_test)
    X_test = df_test[feat_cols].values
    
    print(f"Data shape: {X_test.shape}")
    print(f"Features: {len(feat_cols)}")

    # Initialize Benchmark Suite
    suite = BenchmarkSuite(output_dir="./outputs/benchmarks_v2")
    
    # 1. Performance
    print("Running Inference Latency...")
    suite.benchmark_inference_latency(model_wrapper, X_test[:100])
    
    print("Running Throughput...")
    suite.benchmark_throughput(model_wrapper, X_test[:1000])
    
    # 2. Network & Security (MISSING IN PREVIOUS RUN)
    print("Running Network Resilience Benchmark...")
    suite.benchmark_network_resilience(X_test[:500])
    
    print("Running Encryption Overhead Benchmark...")
    suite.benchmark_encryption(EncryptionManager())
    
    # 3. Quality & Lead Time
    print("Running Detection Quality Benchmark...")
    # Use 100 cycle threshold to ensure we have positive samples in FD001 test set
    suite.benchmark_detection_quality(model_wrapper, X_test, df_test, rul_truth, threshold_cycles=100)
    
    print("Running Lead Time Benchmark...")
    suite.benchmark_lead_time(model_wrapper, X_test, df_test, rul_truth, failure_threshold_cycles=100)
    
    # 4. Robustness
    print("Running Robustness Benchmark...")
    suite.benchmark_robustness(model_wrapper, X_test, df_test, rul_truth, threshold_cycles=100)
    
    # Export
    print("Generating report and plots...")
    suite.save_results()
    suite.generate_report_table()
    try:
        suite.generate_plots()
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nVerification complete. Results saved to {suite.output_dir}")
    print(f"Benchmark Report: {suite.output_dir}/benchmark_report.md")
    print(f"Benchmark Plot: {suite.output_dir}/benchmark_plots.png")

if __name__ == "__main__":
    run_verification()
