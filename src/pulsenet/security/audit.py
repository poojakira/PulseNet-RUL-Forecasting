# pyright: reportGeneralTypeIssues=false
"""
Access audit logging -- tracks who accessed what endpoint, when, with what role.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, cast

from pulsenet.config import cfg
from pulsenet.logger import get_logger

log = get_logger(__name__)
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GENESIS_HASH = "0" * 64


class AuditLogger:
    """Append-only access audit log with hash-chain integrity."""

    def __init__(self, log_file: Optional[str] = None):
        default_log = getattr(cfg.api, "audit_log", "access_audit.jsonl")
        self.log_file = Path(log_file or default_log)

    def _get_log_path(self, tenant_id: str) -> Path:
        """Helper to compute tenant-isolated log path."""
        if not _TENANT_ID_RE.fullmatch(tenant_id):
            raise ValueError("Invalid tenant identifier")
        if tenant_id == "public":
            return self.log_file
        safe_tenant_path = (self.log_file.parent / f"access_audit_{tenant_id}.jsonl").resolve()
        base = self.log_file.parent.resolve()
        if base != safe_tenant_path.parent:
            raise ValueError(f"Path traversal attempt detected: {tenant_id}")
        return safe_tenant_path

    @staticmethod
    def _entry_hash(entry: dict[str, Any]) -> str:
        payload = {k: v for k, v in entry.items() if k != "hash"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _last_chain_state(self, log_path: Path) -> tuple[int, str]:
        if not log_path.exists():
            return 0, _GENESIS_HASH
        last_sequence = -1
        last_hash = _GENESIS_HASH
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = cast(dict[str, Any], json.loads(line))
                except json.JSONDecodeError:
                    continue
                sequence = entry.get("sequence")
                entry_hash = entry.get("hash")
                if isinstance(sequence, int) and isinstance(entry_hash, str):
                    last_sequence = max(last_sequence, sequence)
                    last_hash = entry_hash
        return last_sequence + 1, last_hash

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
        try:
            log_path = self._get_log_path(tenant_id)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sequence, previous_hash = self._last_chain_state(log_path)
            entry: dict[str, Any] = {
                "sequence": sequence,
                "previous_hash": previous_hash,
                "timestamp": time.time(),
                "endpoint": endpoint,
                "method": method,
                "user": user,
                "role": role,
                "status_code": status_code,
                "tenant": tenant_id,
                "metadata": metadata or {},
            }
            entry["hash"] = self._entry_hash(entry)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

            log.debug(
                "Access logged",
                extra={"endpoint": endpoint, "user": user, "role": role},
            )
            return str(entry["hash"])
        except Exception as e:
            log.error(f"Failed to write audit log: {e}")
            return "ACCESS_LOG_FAILURE"

    def get_recent(
        self, n: int = 50, tenant_id: str = "public"
    ) -> list[dict[str, Any]]:
        """Return the last N audit entries for a given tenant."""
        log_path = self._get_log_path(tenant_id)
        if not log_path.exists():
            return []

        try:
            with open(log_path, "r", encoding="utf-8") as f:
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
        """Verify hashes, sequence continuity, and chain links for a tenant."""
        log_path = self._get_log_path(tenant_id)
        if not log_path.exists():
            return True, 0

        corrupt = 0
        expected_sequence = 0
        expected_previous_hash = _GENESIS_HASH
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = cast(dict[str, Any], json.loads(line))
                        stored_hash = entry.get("hash", "")
                        recomputed = self._entry_hash(entry)
                        has_chain_fields = "sequence" in entry or "previous_hash" in entry
                        if stored_hash != recomputed:
                            corrupt += 1
                        if has_chain_fields:
                            if entry.get("sequence") != expected_sequence:
                                corrupt += 1
                            if entry.get("previous_hash") != expected_previous_hash:
                                corrupt += 1
                            expected_sequence += 1
                            expected_previous_hash = str(stored_hash)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        corrupt += 1
            return corrupt == 0, corrupt
        except Exception as e:
            log.error(f"Audit integrity check failed for {tenant_id}: {e}")
            return False, -1