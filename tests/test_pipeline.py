"""
Integration tests for the full pipeline.
"""

from __future__ import annotations

import pytest

from pulsenet.pipeline.preprocessing import (
    compute_rolling_features,
    normalize,
    get_feature_columns,
    create_labels,
    create_sequences,
)


class TestPreprocessing:
    """Tests for preprocessing module."""

    def test_rolling_features(self, sample_sensor_data):
        df = compute_rolling_features(sample_sensor_data.copy(), window=3)
        rolling_cols = [c for c in df.columns if "rolling" in c]
        assert len(rolling_cols) > 0
        for col in rolling_cols:
            assert not df[col].isna().all()

    def test_normalize(self, sample_features):
        feat_cols = get_feature_columns(sample_features)
        df1 = sample_features.copy()
        df2 = sample_features.copy()
        df1, df2, scaler = normalize(df1, df2)
        # After normalization, values should be in [0, 1]
        for col in feat_cols:
            assert df1[col].min() >= -0.01  # allow small float error
            assert df1[col].max() <= 1.01

    def test_get_feature_columns(self, sample_features):
        cols = get_feature_columns(sample_features)
        assert "unit_number" not in cols
        assert "time_in_cycles" not in cols
        assert len(cols) > 0

    def test_create_labels(self, sample_sensor_data, sample_rul):
        labels = create_labels(sample_sensor_data, sample_rul, failure_threshold=30)
        assert len(labels) == len(sample_sensor_data)
        assert set(labels).issubset({0, 1})

    def test_create_sequences(self, sample_features):
        feat_cols = get_feature_columns(sample_features)
        seqs = create_sequences(sample_features, feat_cols, seq_len=10)
        assert seqs.ndim == 3
        assert seqs.shape[1] == 10
        assert seqs.shape[2] == len(feat_cols)


class TestStreamingQueue:
    """Tests for async streaming queue."""

    @pytest.mark.asyncio
    async def test_put_get(self):
        from pulsenet.streaming.queue import AsyncStreamQueue

        q = AsyncStreamQueue(max_size=10)
        await q.put({"sensor": 0.5})
        item = await q.get()
        assert item == {"sensor": 0.5}

    @pytest.mark.asyncio
    async def test_backpressure(self):
        from pulsenet.streaming.queue import AsyncStreamQueue

        q = AsyncStreamQueue(max_size=10, backpressure_threshold=0.5)
        for i in range(6):
            await q.put({"i": i})
        assert q.is_backpressured

    @pytest.mark.asyncio
    async def test_drain_batch(self):
        from pulsenet.streaming.queue import AsyncStreamQueue

        q = AsyncStreamQueue(max_size=100)
        for i in range(20):
            await q.put({"i": i})
        batch = await q.drain_batch(batch_size=10)
        assert len(batch) == 10

    @pytest.mark.asyncio
    async def test_metrics(self):
        from pulsenet.streaming.queue import AsyncStreamQueue

        q = AsyncStreamQueue(max_size=10)
        await q.put({"data": 1})
        await q.get()
        metrics = q.get_metrics()
        assert metrics["enqueued"] == 1
        assert metrics["dequeued"] == 1


class TestConfigAndLogging:
    """Tests for config and logging modules."""

    def test_config_loads(self):
        from pulsenet.config import cfg

        assert hasattr(cfg, "system") or cfg.__class__.__name__ == "SimpleNamespace"

    def test_logger(self):
        from pulsenet.logger import get_logger

        log = get_logger("test_logger", level="DEBUG", fmt="text")
        log.info("Test message")  # Should not raise
