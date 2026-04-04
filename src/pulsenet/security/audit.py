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

    def _get_log_path(self, tenant_id: str) -> Path:
        """Helper to compute tenant-isolated log path."""
        # Special case: if log_file is a .jsonl file, use it directly for 'public'
        if tenant_id == "public":
            return self.log_file
        return self.log_file.parent / f"access_audit_{tenant_id}.jsonl"

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

            log_path = self._get_log_path(tenant_id)
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

    def get_recent(self, n: int = 50, tenant_id: str = "public") -> list[dict[str, Any]]:
        """Return the last N audit entries for a given tenant."""
        log_path = self._get_log_path(tenant_id)
        if not log_path.exists():
            return []

        try:
            with open(log_path, "r") as f:
                lines = f.readlines()

            entries: list[dict[str, Any]] = []
            for line in lines[-n:]:
                try:
                    entries.append(cast(dict[str, Any], json.loads(line)))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception as e:
            log.warning(f"Failed to read recent audit entries for {tenant_id}: {e}")
            return []

    def verify_integrity(self, tenant_id: str = "public") -> tuple[bool, int]:
        """Verify hash integrity of all entries for a given tenant."""
        log_path = self._get_log_path(tenant_id)
        if not log_path.exists():
            return True, 0

        corrupt = 0
        try:
            with open(log_path, "r") as f:
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
            log.error(f"Audit integrity check failed for {tenant_id}: {e}")
            return False, -1
