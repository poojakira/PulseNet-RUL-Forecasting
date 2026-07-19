"""Tests for MLOps tracking, streaming producer/consumer, and async queue."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from pulsenet.mlops.tracker import MLOpsTracker
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.streaming.consumer import InferenceConsumer
from pulsenet.streaming.producer import SensorProducer
from pulsenet.streaming.queue import AsyncStreamQueue


class TestMLOpsTracker:
    def test_set_reference_and_detect_drift(self, tmp_path):
        t = MLOpsTracker(tracking_uri=str(tmp_path / "mlruns"), drift_threshold=0.1)
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((200, 5))
        t.set_reference_distribution(ref)

        no_drift = t.detect_drift(ref + 0.001)
        assert no_drift["drift_detected"] is False

        drift = t.detect_drift(rng.standard_normal((200, 5)) * 6 + 20)
        assert drift["drift_detected"] is True
        assert drift["retrain_recommended"] is True

    def test_detect_drift_without_reference(self, tmp_path):
        t = MLOpsTracker(tracking_uri=str(tmp_path / "mlruns"))
        result = t.detect_drift(np.zeros((5, 3)))
        assert result["drift_detected"] is False

    def test_log_training_run_local_fallback(self, tmp_path):
        t = MLOpsTracker(tracking_uri=str(tmp_path / "mlruns"))
        t._mlflow_available = False
        digest = t.log_training_run({"n_estimators": 100}, {"f1": 0.87})
        assert isinstance(digest, str) and len(digest) == 64
        assert (tmp_path / "mlruns" / "local_tracking.jsonl").exists()

    def test_log_inference_metrics(self, tmp_path):
        t = MLOpsTracker(tracking_uri=str(tmp_path / "mlruns"))
        t._mlflow_available = False
        # Should not raise
        t.log_inference_metrics(np.array([0, 1, 1, 0]), latency_ms=12.3, batch_size=4)

    def test_log_training_run_mlflow_backend(self, tmp_path):
        t = MLOpsTracker(tracking_uri=str(tmp_path / "mlruns"))
        if t._mlflow_available:
            run_id = t.log_training_run({"p": 1}, {"m": 0.9})
            assert isinstance(run_id, str) and run_id


class TestStreaming:
    async def test_producer_streams_rows(self, tmp_path):
        csv = tmp_path / "feed.csv"
        pd.DataFrame(
            {
                "unit_number": [1, 1, 1],
                "time_in_cycles": [1, 2, 3],
                "sensor_2": [0.1, 0.2, 0.3],
            }
        ).to_csv(csv, index=False)

        q = AsyncStreamQueue(max_size=100)
        producer = SensorProducer(q, data_path=str(csv), delay_ms=0, loop=False)
        await producer.start()
        assert producer.metrics["produced"] == 3
        assert q.size == 3

    async def test_producer_missing_file(self, tmp_path):
        q = AsyncStreamQueue()
        producer = SensorProducer(q, data_path=str(tmp_path / "nope.csv"))
        await producer.start()
        assert producer.metrics["produced"] == 0

    async def test_consumer_process_batch(self):
        rng = np.random.default_rng(1)
        model = IsolationForestModel(n_estimators=20)
        model.train(rng.standard_normal((40, 3)))

        q = AsyncStreamQueue()
        consumer = InferenceConsumer(
            q, model, feature_cols=["a", "b", "c"], batch_size=8
        )
        batch = [
            {
                "a": float(i),
                "b": float(-i),
                "c": float(i % 3),
                "unit_number": 1,
                "time_in_cycles": i,
            }
            for i in range(10)
        ]
        await consumer._process_batch(batch)
        assert consumer.metrics["processed"] == 10
        consumer.stop()
        assert consumer.metrics["running"] is False

    async def test_consumer_start_and_stop_loop(self):
        rng = np.random.default_rng(2)
        model = IsolationForestModel(n_estimators=20)
        model.train(rng.standard_normal((30, 3)))

        q = AsyncStreamQueue()
        consumer = InferenceConsumer(
            q, model, feature_cols=["a", "b", "c"], batch_size=4
        )
        for i in range(4):
            await q.put(
                {"a": 1.0, "b": 2.0, "c": 3.0, "unit_number": 1, "time_in_cycles": i}
            )

        task = asyncio.create_task(consumer.start())
        await asyncio.sleep(0.3)
        consumer.stop()
        await asyncio.sleep(0.2)
        if not task.done():
            task.cancel()
        assert consumer.metrics["processed"] >= 4


class TestAsyncStreamQueue:
    async def test_put_get_and_metrics(self):
        q = AsyncStreamQueue(max_size=10)
        assert await q.put({"x": 1}) is True
        assert q.size == 1
        item = await q.get(timeout=1.0)
        assert item == {"x": 1}
        metrics = q.get_metrics()
        assert metrics["enqueued"] == 1
        assert metrics["dequeued"] == 1

    async def test_drain_batch(self):
        q = AsyncStreamQueue(max_size=10)
        for i in range(5):
            await q.put({"i": i})
        batch = await q.drain_batch(batch_size=3)
        assert len(batch) == 3
        assert q.is_backpressured is False

    async def test_get_timeout_returns_none(self):
        q = AsyncStreamQueue()
        assert await q.get(timeout=0.05) is None

    async def test_backpressure_flag(self):
        q = AsyncStreamQueue(max_size=4, backpressure_threshold=0.5)
        for i in range(3):
            await q.put({"i": i})
        # size 3 / 4 > 0.5 -> backpressured; next put exercises the warning path
        assert q.is_backpressured is True
        await q.put({"i": 99})
        assert q.get_metrics()["backpressure_events"] >= 1

    async def test_put_drops_when_full(self):
        q = AsyncStreamQueue(max_size=1)
        assert await q.put({"a": 1}) is True
        # Queue full -> put times out quickly and drops the item
        assert await q.put({"b": 2}, timeout=0.05) is False
        assert q.get_metrics()["dropped"] == 1

    async def test_drain_batch_empty_waits_then_returns(self):
        q = AsyncStreamQueue()
        batch = await q.drain_batch(batch_size=3, timeout=0.05)
        assert batch == []
