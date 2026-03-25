"""
Access audit logging — tracks who accessed what endpoint, when, with what role.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from pulsenet.logger import get_logger

log = get_logger(__name__)


class AuditLogger:
    """Append-only access audit log with hash integrity."""

    def __init__(self, log_file: str = "access_audit.jsonl"):
        self.log_file = Path(log_file)

    def log_access(
        self,
        endpoint: str,
        method: str,
        user: str = "anonymous",
        role: str = "unknown",
        status_code: int = 200,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record an access event. Returns the entry hash."""
        entry = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "method": method,
            "user": user,
            "role": role,
            "status_code": status_code,
            "metadata": metadata or {},
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["hash"] = entry_hash

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        log.debug("Access logged", extra={"endpoint": endpoint, "user": user, "role": role})
        return entry_hash

    def get_recent(self, n: int = 50) -> list[dict]:
        """Return the last N audit entries."""
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def verify_integrity(self) -> tuple[bool, int]:
        """Verify hash integrity of all entries.

        Returns (all_valid, corrupt_count).
        """
        if not self.log_file.exists():
            return True, 0
        corrupt = 0
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    stored_hash = entry.pop("hash", "")
                    recomputed = hashlib.sha256(
                        json.dumps(entry, sort_keys=True).encode()
                    ).hexdigest()
                    if stored_hash != recomputed:
                        corrupt += 1
                except (json.JSONDecodeError, KeyError):
                    corrupt += 1
        return corrupt == 0, corrupt
