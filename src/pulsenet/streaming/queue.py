"""
Async queue with backpressure support for sensor data streaming.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from pulsenet.logger import get_logger

log = get_logger(__name__)


class AsyncStreamQueue:
    """Bounded async queue with backpressure and batch drain."""

    def __init__(self, max_size: int = 1000, backpressure_threshold: float = 0.8):
        self.max_size = max_size
        self.backpressure_threshold = backpressure_threshold
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._metrics = {
            "enqueued": 0,
            "dequeued": 0,
            "dropped": 0,
            "backpressure_events": 0,
        }

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_backpressured(self) -> bool:
        return self.size / self.max_size > self.backpressure_threshold

    async def put(self, item: Any, timeout: float = 5.0) -> bool:
        """Enqueue an item. Returns False if queue is full after timeout."""
        if self.is_backpressured:
            self._metrics["backpressure_events"] += 1
            log.warning("Backpressure active",
                       extra={"queue_size": self.size, "max": self.max_size})

        try:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
            self._metrics["enqueued"] += 1
            return True
        except asyncio.TimeoutError:
            self._metrics["dropped"] += 1
            log.warning("Item dropped — queue full")
            return False

    async def get(self, timeout: float = 5.0) -> Optional[Any]:
        """Dequeue a single item."""
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._metrics["dequeued"] += 1
            return item
        except asyncio.TimeoutError:
            return None

    async def drain_batch(self, batch_size: int = 32, timeout: float = 1.0) -> list[Any]:
        """Drain up to batch_size items from the queue."""
        batch = []
        for _ in range(batch_size):
            try:
                item = self._queue.get_nowait()
                batch.append(item)
                self._metrics["dequeued"] += 1
            except asyncio.QueueEmpty:
                if not batch:
                    # Wait for at least one item
                    item = await self.get(timeout=timeout)
                    if item is not None:
                        batch.append(item)
                break
        return batch

    def get_metrics(self) -> dict:
        return {**self._metrics, "current_size": self.size, "utilization": self.size / self.max_size}
