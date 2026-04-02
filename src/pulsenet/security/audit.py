# pyright: reportGeneralTypeIssues=false
"""
Access audit logging — tracks who accessed what endpoint, when, with what role.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, cast

from pulsenet.config import cfg
from pulsenet.logger import get_logger

log = get_logger(__name__)


class AuditLogger:
    """Append-only access audit log with hash integrity."""

    def __init__(self, log_file: Optional[str] = None):
        # Use config as default
        default_log = getattr(cfg.api, "audit_log", "access_audit.jsonl")
        self.log_file = Path(log_file or default_log)

    def log_access(
        self,
        endpoint: str,
        method: str,
        user: str = "anonymous",
        role: str = "unknown",
        status_code: int = 200,
        metadata: Optional[dict[str, Any]] = None,
        tenant_id: str = "public",
    ) -> str:
        """Record an access event for a specific tenant. Returns the entry hash."""
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "method": method,
            "user": user,
            "role": role,
            "status_code": status_code,
            "tenant": tenant_id,
            "metadata": metadata or {},
        }

        try:
            entry_str = json.dumps(entry, sort_keys=True)
            entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
            entry["hash"] = entry_hash

            # Tenant-isolated file naming
            log_path = self.log_file.parent / f"access_audit_{tenant_id}.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            log.debug(
                "Access logged",
                extra={"endpoint": endpoint, "user": user, "role": role},
            )
            return entry_hash
        except Exception as e:
            log.error(f"Failed to write audit log: {e}")
            return "ACCESS_LOG_FAILURE"

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last N audit entries."""
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()

            entries: list[dict[str, Any]] = []
            for line in lines[-n:]:
                try:
                    entries.append(cast(dict[str, Any], json.loads(line)))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception as e:
            log.warning(f"Failed to read recent audit entries: {e}")
            return []

    def verify_integrity(self) -> tuple[bool, int]:
        """Verify hash integrity of all entries."""
        if not self.log_file.exists():
            return True, 0

        corrupt = 0
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entry = cast(dict[str, Any], json.loads(line))
                        stored_hash = entry.pop("hash", "")
                        recomputed = hashlib.sha256(
                            json.dumps(entry, sort_keys=True).encode()
                        ).hexdigest()
                        if stored_hash != recomputed:
                            corrupt += 1
                    except (json.JSONDecodeError, KeyError):
                        corrupt += 1
            return corrupt == 0, corrupt
        except Exception as e:
            log.error(f"Audit integrity check failed: {e}")
            return False, -1
