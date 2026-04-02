"""
ML pipeline consumer — drains batches from queue, runs inference, logs results.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import pandas as pd

from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel
from pulsenet.security.blockchain import BlackBoxLedger
from pulsenet.streaming.queue import AsyncStreamQueue

log = get_logger(__name__)


class InferenceConsumer:
    """Consumes sensor batches, runs ML inference, logs to blockchain."""

    def __init__(
        self,
        queue: AsyncStreamQueue,
        model: BaseAnomalyModel,
        ledger: Optional[BlackBoxLedger] = None,
        feature_cols: Optional[list[str]] = None,
        batch_size: int = 32,
    ):
        self.queue = queue
        self.model = model
        self.ledger = ledger or BlackBoxLedger()
        self.feature_cols = feature_cols or []
        self.batch_size = batch_size
        self._running = False
        self._processed = 0
        self._anomalies = 0
        self._latencies: list[float] = []

    async def start(self) -> None:
        """Begin consuming and processing sensor readings."""
        self._running = True
        log.info("Consumer started", extra={"batch_size": self.batch_size})

        while self._running:
            batch = await self.queue.drain_batch(self.batch_size)
            if not batch:
                await asyncio.sleep(0.1)
                continue

            await self._process_batch(batch)

        log.info(
            "Consumer stopped",
            extra={"processed": self._processed, "anomalies": self._anomalies},
        )

    async def _process_batch(self, batch: list[dict]) -> None:
        """Process a batch of sensor readings."""
        t0 = time.perf_counter()

        df = pd.DataFrame(batch)
        if self.feature_cols:
            feat_cols = [c for c in self.feature_cols if c in df.columns]
        else:
            feat_cols = [
                c
                for c in df.columns
                if c not in ("unit_number", "time_in_cycles", "is_anomaly")
            ]

        if not feat_cols:
            return

        X = df[feat_cols]
        predictions = self.model.predict(X)
        scores = self.model.score(X)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(latency_ms)

        n_anomalies = int(predictions.sum())
        self._processed += len(batch)
        self._anomalies += n_anomalies

        # Log critical events to blockchain
        for i, pred in enumerate(predictions):
            if pred == 1:
                row = batch[i]
                self.ledger.add_entry(
                    unit_id=int(row.get("unit_number", 0)),
                    cycles=int(row.get("time_in_cycles", 0)),
                    health_score=max(0, min(100, (1 - float(scores[i])) * 100)),
                    status="CRITICAL",
                )

        if self._processed % 100 == 0:
            log.info(
                f"Processed {self._processed} readings",
                extra={
                    "anomalies_total": self._anomalies,
                    "batch_latency_ms": round(latency_ms, 2),
                    "avg_latency_ms": round(
                        sum(self._latencies) / len(self._latencies), 2
                    ),
                },
            )

    def stop(self) -> None:
        self._running = False

    @property
    def metrics(self) -> dict:
        avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        return {
            "processed": self._processed,
            "anomalies": self._anomalies,
            "avg_latency_ms": round(avg_lat, 2),
            "running": self._running,
        }
