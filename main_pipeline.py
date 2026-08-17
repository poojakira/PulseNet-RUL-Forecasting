"""PulseNet CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pulsenet.config import cfg
from pulsenet.logger import get_logger

log = get_logger("pulsenet.cli")


def run_full_pipeline() -> None:
    """Execute ingestion, preprocessing, training, evaluation, and inference."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator

    pipeline = PipelineOrchestrator(data_dir=cfg.system.data_dir)
    results = pipeline.run_full_pipeline()

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        if isinstance(metrics, dict) and "error" not in metrics:
            for key, value in metrics.items():
                print(
                    f"  {key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"  {key}: {value}"
                )
        else:
            print(f"  {metrics}")
    print("=" * 60)


def run_training() -> None:
    """Train the configured model."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator

    pipeline = PipelineOrchestrator(data_dir=cfg.system.data_dir)
    pipeline.run_ingestion()
    pipeline.run_preprocessing()
    pipeline.run_training()
    print("[SUCCESS] Training complete")


def run_prediction() -> None:
    """Run inference on official FD001 test data."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator

    pipeline = PipelineOrchestrator(data_dir=cfg.system.data_dir)
    pipeline.run_ingestion()
    pipeline.run_preprocessing()
    pipeline.run_training()
    result_df = pipeline.run_inference()
    anomalies = result_df["prediction"].sum()
    print(
        f"[SUCCESS] Inference complete: {anomalies}/{len(result_df)} anomalies detected"
    )


def run_benchmark() -> None:
    """Run official-data validation metrics and evidence generation."""
    from scripts.run_validation import main as run_validation_main

    run_validation_main()


def run_streaming() -> None:
    """Start async streaming from an approved telemetry file."""

    async def _stream() -> None:
        from pulsenet.models.isolation_forest import IsolationForestModel
        from pulsenet.security.blockchain import BlackBoxLedger
        from pulsenet.streaming.consumer import InferenceConsumer
        from pulsenet.streaming.producer import SensorProducer
        from pulsenet.streaming.queue import AsyncStreamQueue

        model = IsolationForestModel()
        model_path = Path("models/isolation_forest.joblib")
        if not model_path.exists():
            print("[WARNING] No trained model found. Run --mode train first.")
            return

        model.load(model_path)
        queue = AsyncStreamQueue(max_size=1000)
        producer = SensorProducer(
            queue, data_path="data/test_features.csv", delay_ms=30
        )
        consumer = InferenceConsumer(queue, model, BlackBoxLedger(), batch_size=32)

        print("[INFO] Streaming pipeline started (Ctrl+C to stop)")
        try:
            await asyncio.gather(producer.start(), consumer.start())
        except KeyboardInterrupt:
            producer.stop()
            consumer.stop()
            print("\n[METRICS] Final metrics:")
            print(f"  Producer: {producer.metrics}")
            print(f"  Consumer: {consumer.metrics}")
            print(f"  Queue: {queue.get_metrics()}")

    asyncio.run(_stream())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PulseNet Predictive Maintenance Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "predict", "benchmark", "stream"],
        default="full",
        help="Pipeline execution mode",
    )
    args = parser.parse_args()

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
