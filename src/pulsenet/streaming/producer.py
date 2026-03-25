"""
Sensor data stream producer — reads CSV data and pushes to async queue.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from pulsenet.streaming.queue import AsyncStreamQueue
from pulsenet.logger import get_logger

log = get_logger(__name__)


class SensorProducer:
    """Simulates real-time sensor telemetry by streaming CSV rows."""

    def __init__(
        self,
        queue: AsyncStreamQueue,
        data_path: str = "test_features.csv",
        delay_ms: float = 30,
        loop: bool = False,
    ):
        self.queue = queue
        self.data_path = Path(data_path)
        self.delay_ms = delay_ms
        self.loop = loop
        self._running = False
        self._produced = 0

    async def start(self) -> None:
        """Begin producing sensor readings to the queue."""
        if not self.data_path.exists():
            log.error(f"Data file not found: {self.data_path}")
            return

        df = pd.read_csv(self.data_path)
        self._running = True
        log.info("Producer started",
                extra={"rows": len(df), "delay_ms": self.delay_ms})

        while self._running:
            for _, row in df.iterrows():
                if not self._running:
                    break

                await self.queue.put(row.to_dict())
                self._produced += 1

                if self._produced % 100 == 0:
                    log.debug(f"Produced {self._produced} readings",
                             extra={"queue_size": self.queue.size})

                await asyncio.sleep(self.delay_ms / 1000)

            if not self.loop:
                break

        log.info("Producer stopped", extra={"total_produced": self._produced})

    def stop(self) -> None:
        self._running = False

    @property
    def metrics(self) -> dict:
        return {"produced": self._produced, "running": self._running}
