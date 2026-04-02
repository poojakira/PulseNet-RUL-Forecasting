# pyright: reportGeneralTypeIssues=false
"""
Blockchain audit ledger with Merkle tree optimization.

Provides tamper-proof audit logging for maintenance events using
SHA-256 hash chaining and optional Merkle root computation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from pulsenet.logger import get_logger

log = get_logger(__name__)


class _NpEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):  # type: ignore
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):  # type: ignore
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

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Compute the SHA-256 hash of the block contents."""
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

    def to_dict(self) -> dict[str, Any]:
        """Convert block to a serializable dictionary."""
        return cast(dict[str, Any], asdict(self))


class BlackBoxLedger:
    """Cryptographic ledger recording maintenance & anomaly events."""

    def __init__(self, base_path: Optional[str] = None, enable_merkle: bool = True):
        # Use config for storage path
        env_ledger = os.environ.get("PULSENET_LEDGER_PATH", "ledger.json")
        self.base_path = Path(base_path or env_ledger).parent
        self.enable_merkle = enable_merkle
        self.tenants: dict[str, list[Block]] = {}
        self._metrics: dict[str, Any] = {"total_blocks": 0, "avg_add_latency_ms": 0.0}
        self._latencies: list[float] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core chain operations
    # ------------------------------------------------------------------
    def _create_genesis_block(self, tenant_id: str = "public") -> None:
        """Initialize a tenant's chain with a genesis block."""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data=f"GENESIS_BLOCK_TENANT_{tenant_id.upper()}",
            previous_hash="0",
        )
        self.tenants[tenant_id] = [genesis]
        self.save_chain(tenant_id)
        log.info(f"Genesis block created for tenant: {tenant_id}", extra={"hash": genesis.hash[:16]})

    def add_entry(
        self,
        unit_id: int,
        cycles: int,
        health_score: float,
        status: str,
        tenant_id: str = "public",
    ) -> str:
        """Add a new maintenance event block for a specific tenant."""
        t0 = time.perf_counter()
        with self.lock:
            if tenant_id not in self.tenants:
                self.load_chain(tenant_id) # Try to load or create
            
            chain = self.tenants[tenant_id]
            previous = chain[-1]
            data_payload = {
                "unit_id": unit_id,
                "cycles": cycles,
                "health_score": round(health_score, 2),
                "status": status,
                "tenant": tenant_id
            }
            new_block = Block(
                index=previous.index + 1,
                timestamp=time.time(),
                data=data_payload,
                previous_hash=previous.hash,
            )
            chain.append(new_block)

            # Flush to disk every 10 blocks or on critical status
            if len(chain) % 10 == 0 or status == "CRITICAL":
                self.save_chain(tenant_id)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(latency_ms)
        self._metrics["total_blocks_global"] = sum(len(c) for c in self.tenants.values())
        self._metrics["avg_add_latency_ms"] = sum(self._latencies) / len(
            self._latencies
        )

        return new_block.hash

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_integrity(self, tenant_id: str = "public") -> tuple[bool, str]:
        """Verify the entire chain of a tenant. Returns (is_valid, message)."""
        with self.lock:
            if tenant_id not in self.tenants:
                self.load_chain(tenant_id)
            chain = self.tenants[tenant_id]
            for i in range(1, len(chain)):
                current = chain[i]
                previous = chain[i - 1]
                if current.hash != current.calculate_hash():
                    return False, f"Block #{current.index} data tampered!"
                if current.previous_hash != previous.hash:
                    return False, f"Broken chain link at Block #{current.index}"
        return True, "Ledger integrity verified"

    def detect_tampering(self, tenant_id: str = "public") -> list[int]:
        """Return indices of tampered blocks in a tenant's chain."""
        tampered: list[int] = []
        with self.lock:
            if tenant_id not in self.tenants:
                self.load_chain(tenant_id)
            chain = self.tenants[tenant_id]
            for i in range(1, len(chain)):
                blk = chain[i]
                if blk.hash != blk.calculate_hash():
                    tampered.append(blk.index)
                if blk.previous_hash != chain[i - 1].hash:
                    tampered.append(blk.index)
        return list(set(tampered))

    # ------------------------------------------------------------------
    # Merkle tree
    # ------------------------------------------------------------------
    def compute_merkle_root(self, tenant_id: str = "public") -> str:
        """Compute Merkle root hash over all blocks of a tenant."""
        with self.lock:
            if tenant_id not in self.tenants:
                self.load_chain(tenant_id)
            chain = self.tenants[tenant_id]
            if not chain:
                return hashlib.sha256(b"empty").hexdigest()
            hashes = [b.hash for b in chain]

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
    def save_chain(self, tenant_id: str = "public") -> None:
        """Persist a tenant's chain to disk."""
        try:
            chain = self.tenants[tenant_id]
            chain_data = [b.to_dict() for b in chain]
            storage_path = self.base_path / f"ledger_{tenant_id}.json"
            with open(storage_path, "w") as f:
                json.dump(chain_data, f, indent=2, cls=_NpEncoder)
        except Exception as e:
            log.error(f"Failed to save blockchain ledger for tenant {tenant_id}: {e}")

    def load_chain(self, tenant_id: str = "public") -> None:
        """Load a tenant's chain from disk."""
        storage_path = self.base_path / f"ledger_{tenant_id}.json"
        if not storage_path.exists():
            self._create_genesis_block(tenant_id)
            return

        try:
            with open(storage_path, "r") as f:
                chain_data = json.load(f)
                self.tenants[tenant_id] = [Block(**d) for d in chain_data]
            log.info(f"Chain loaded for tenant {tenant_id}", extra={"blocks": len(self.tenants[tenant_id])})
        except Exception as e:
            log.error(f"Failed to load blockchain ledger for tenant {tenant_id}: {e}")
            self._create_genesis_block(tenant_id)

    # ------------------------------------------------------------------
    # Metrics / API helpers
    # ------------------------------------------------------------------
    def get_metrics(self, tenant_id: str = "public") -> dict[str, Any]:
        """Return system health and integrity metrics for a tenant."""
        return {
            **self._metrics,
            "merkle_root": self.compute_merkle_root(tenant_id) if self.enable_merkle else None,
            "chain_valid": self.validate_integrity(tenant_id)[0],
        }

    def get_recent_blocks(self, n: int = 10, tenant_id: str = "public") -> list[dict[str, Any]]:
        """Return the last N blocks of a tenant."""
        with self.lock:
            if tenant_id not in self.tenants:
                self.load_chain(tenant_id)
            return [b.to_dict() for b in self.tenants[tenant_id][-n:]]
