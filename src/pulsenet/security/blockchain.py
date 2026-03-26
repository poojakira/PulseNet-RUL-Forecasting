"""
Blockchain audit ledger with Merkle tree optimization.

Provides tamper-proof audit logging for maintenance events using
SHA-256 hash chaining and optional Merkle root computation.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from pulsenet.logger import get_logger

log = get_logger(__name__)


class _NpEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class Block:
    """Single block in the audit chain."""

    index: int
    timestamp: float
    data: Any
    previous_hash: str
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": self.data,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
            cls=_NpEncoder,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


class BlackBoxLedger:
    """Cryptographic ledger recording maintenance & anomaly events."""

    def __init__(self, storage_path: str = "ledger.json", enable_merkle: bool = True):
        self.storage_path = Path(storage_path)
        self.enable_merkle = enable_merkle
        self.chain: list[Block] = []
        self._metrics = {"total_blocks": 0, "avg_add_latency_ms": 0.0}
        self._latencies: list[float] = []
        self.lock = threading.Lock()

        # Load or init
        if self.storage_path.exists():
            self.load_chain()
        else:
            self._create_genesis_block()

    # ------------------------------------------------------------------
    # Core chain operations
    # ------------------------------------------------------------------
    def _create_genesis_block(self) -> None:
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data="GENESIS_BLOCK_ENGINE_START",
            previous_hash="0",
        )
        self.chain.append(genesis)
        self.save_chain()
        log.info("Genesis block created", extra={"hash": genesis.hash[:16]})

    def add_entry(
        self,
        unit_id: int,
        cycles: int,
        health_score: float,
        status: str,
    ) -> str:
        """Add a new maintenance event block. Returns the new block hash."""
        t0 = time.perf_counter()
        with self.lock:
            previous = self.chain[-1]
            data_payload = {
                "unit_id": unit_id,
                "cycles": cycles,
                "health_score": round(health_score, 2),
                "status": status,
            }
            new_block = Block(
                index=previous.index + 1,
                timestamp=time.time(),
                data=data_payload,
                previous_hash=previous.hash,
            )
            self.chain.append(new_block)

            # Flush to disk every 10 blocks instead of every single block to prevent I/O bottleneck
            if len(self.chain) % 10 == 0:
                self.save_chain()

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(latency_ms)
        self._metrics["avg_add_latency_ms"] = sum(self._latencies) / len(
            self._latencies
        )
        return new_block.hash
        self._latencies.append(latency_ms)
        self._metrics["total_blocks"] = len(self.chain)
        self._metrics["avg_add_latency_ms"] = sum(self._latencies) / len(
            self._latencies
        )
        # The original add_entry returned new_block.hash, but the new one returns None.
        # Sticking to the provided return type `None`.

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_integrity(self) -> tuple[bool, str]:
        """Verify the entire chain. Returns (is_valid, message)."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.calculate_hash():
                return False, f"Block #{current.index} data tampered!"
            if current.previous_hash != previous.hash:
                return False, f"Broken chain link at Block #{current.index}"
        return True, "Ledger integrity verified"

    def detect_tampering(self) -> list[int]:
        """Return indices of tampered blocks."""
        tampered: list[int] = []
        for i in range(1, len(self.chain)):
            blk = self.chain[i]
            if blk.hash != blk.calculate_hash():
                tampered.append(blk.index)
            if blk.previous_hash != self.chain[i - 1].hash:
                tampered.append(blk.index)
        return list(set(tampered))

    # ------------------------------------------------------------------
    # Merkle tree
    # ------------------------------------------------------------------
    def compute_merkle_root(self) -> str:
        """Compute Merkle root hash over all blocks."""
        if not self.chain:
            return hashlib.sha256(b"empty").hexdigest()
        hashes = [b.hash for b in self.chain]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # duplicate last for odd count
            next_level: list[str] = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level
        return hashes[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_chain(self) -> None:
        chain_data = [b.to_dict() for b in self.chain]
        with open(self.storage_path, "w") as f:
            json.dump(chain_data, f, indent=2, cls=_NpEncoder)

    def load_chain(self) -> None:
        if not self.storage_path.exists():
            return
        with open(self.storage_path, "r") as f:
            chain_data = json.load(f)
            self.chain = [Block(**d) for d in chain_data]
        log.info("Chain loaded", extra={"blocks": len(self.chain)})

    # ------------------------------------------------------------------
    # Metrics / API helpers
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        return {
            **self._metrics,
            "merkle_root": self.compute_merkle_root() if self.enable_merkle else None,
            "chain_valid": self.validate_integrity()[0],
        }

    def get_recent_blocks(self, n: int = 10) -> list[dict]:
        return [b.to_dict() for b in self.chain[-n:]]
