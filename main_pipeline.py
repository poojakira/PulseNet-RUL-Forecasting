"""
PulseNet CLI Orchestrator — central entry point for all pipeline operations.

Usage:
    python main_pipeline.py --mode full        # End-to-end pipeline
    python main_pipeline.py --mode train       # Train models only
    python main_pipeline.py --mode predict     # Run inference only
    python main_pipeline.py --mode benchmark   # Run benchmarks
    python main_pipeline.py --mode stream      # Start streaming pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pulsenet.logger import get_logger

log = get_logger("pulsenet.cli")


def run_full_pipeline():
    """Execute the complete pipeline: ingest → preprocess → train → evaluate → inference."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator
    pipeline = PipelineOrchestrator()
    results = pipeline.run_full_pipeline()

    print("\n" + "=" * 60)
    print("  PIPELINE RESULTS")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n  Model: {model_name}")
        if isinstance(metrics, dict) and "error" not in metrics:
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"    {metrics}")
    print("=" * 60)


def run_training():
    """Train models only (assumes data is already preprocessed)."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator
    pipeline = PipelineOrchestrator()
    pipeline.run_ingestion()
    pipeline.run_preprocessing()
    pipeline.run_training()
    print("✅ Training complete")


def run_prediction():
    """Run inference on test data."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator
    pipeline = PipelineOrchestrator()
    pipeline.run_ingestion()
    pipeline.run_preprocessing()
    pipeline.run_training()
    result_df = pipeline.run_inference()
    anomalies = result_df["prediction"].sum()
    print(f"✅ Inference complete: {anomalies}/{len(result_df)} anomalies detected")


def run_benchmark():
    """Run performance benchmarks."""
    import numpy as np
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator
    from pulsenet.benchmarks.benchmark import BenchmarkSuite
    from pulsenet.security.encryption import EncryptionManager

    pipeline = PipelineOrchestrator()
    pipeline.run_ingestion()
    pipeline.run_preprocessing()
    pipeline.run_training()

    from pulsenet.pipeline.preprocessing import get_feature_columns
    feat_cols = get_feature_columns(pipeline.test_df)
    X = pipeline.test_df[feat_cols].values

    model = pipeline.registry.get_model("isolation_forest")
    bench = BenchmarkSuite()

    bench.benchmark_inference_latency(model, X)
    bench.benchmark_throughput(model, X)
    bench.benchmark_network_resilience(X)
    bench.benchmark_encryption(EncryptionManager())

    try:
        bench.profile_resources()
    except ImportError:
        log.warning("psutil not installed — skipping resource profiling")

    bench.save_results()
    report = bench.generate_report_table()
    bench.generate_plots()

    print("\n" + report)
    print("\n✅ Benchmarks saved to outputs/benchmarks/")


def run_streaming():
    """Start async streaming pipeline (producer + consumer)."""

    async def _stream():
        from pulsenet.streaming.queue import AsyncStreamQueue
        from pulsenet.streaming.producer import SensorProducer
        from pulsenet.streaming.consumer import InferenceConsumer
        from pulsenet.models.isolation_forest import IsolationForestModel
        from pulsenet.security.blockchain import BlackBoxLedger

        # Load model
        model = IsolationForestModel()
        model_path = Path("models/isolation_forest.joblib")
        if not model_path.exists():
            model_path = Path("isolation_forest_model.joblib")
        if model_path.exists():
            model.load(model_path)
        else:
            print("⚠️ No trained model found. Run --mode train first.")
            return

        queue = AsyncStreamQueue(max_size=1000)
        producer = SensorProducer(queue, delay_ms=30)
        consumer = InferenceConsumer(queue, model, BlackBoxLedger(), batch_size=32)

        print("⚡ Streaming pipeline started (Ctrl+C to stop)")
        try:
            await asyncio.gather(
                producer.start(),
                consumer.start(),
            )
        except KeyboardInterrupt:
            producer.stop()
            consumer.stop()
            print("\n📊 Final metrics:")
            print(f"  Producer: {producer.metrics}")
            print(f"  Consumer: {consumer.metrics}")
            print(f"  Queue: {queue.get_metrics()}")

    asyncio.run(_stream())


def main():
    parser = argparse.ArgumentParser(
        description="PulseNet Predictive Maintenance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full       Run complete pipeline (ingest → train → evaluate → inference)
  train      Train models only
  predict    Run inference on test data
  benchmark  Run performance benchmarks
  stream     Start real-time streaming pipeline
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "predict", "benchmark", "stream"],
        default="full",
        help="Pipeline execution mode",
    )

    args = parser.parse_args()

    print(f"\n⚡ PulseNet v2.0 — Mode: {args.mode.upper()}")
    print("=" * 50)

    dispatch = {
        "full": run_full_pipeline,
        "train": run_training,
        "predict": run_prediction,
        "benchmark": run_benchmark,
        "stream": run_streaming,
    }

    dispatch[args.mode]()


if __name__ == "__main__":
    main()
