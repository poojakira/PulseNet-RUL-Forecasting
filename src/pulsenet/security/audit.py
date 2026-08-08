"""
src/pulsenet/security/audit.py
──────────────────────────────────────────────────────────────────────────────
Append-only, SHA-256 hash-chained audit log for PulseNet.

Design
------
Each log entry is a JSON object written as a single newline-delimited line.
The ``previous_hash`` field contains the SHA-256 digest of the *previous*
entry's canonical JSON representation, creating a tamper-evident chain.

The first entry (index 0) carries a sentinel ``previous_hash`` of
``"0" * 64`` (all-zero hex string), referred to as the genesis hash.

Modifying, inserting, or deleting any entry breaks the chain from that
point forward.  ``verify_audit_log()`` replays the full chain and reports
each broken link.

Thread safety
-------------
A module-level ``threading.Lock`` serialises concurrent writes.  Reads
(including verification) acquire no lock and tolerate concurrent writers
by reading a consistent snapshot of whatever is on disk at call time.

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Module-level write lock — ensures atomic append on Windows (no O_APPEND guarantee).
_WRITE_LOCK = threading.Lock()

# Sentinel hash used as ``previous_hash`` for the genesis (first) entry.
_GENESIS_HASH = "0" * 64


class AuditIntegrityError(Exception):
    """Raised when the hash-chain integrity of an audit log is violated.

    Attributes
    ----------
    violations:
        List of human-readable descriptions of each broken link or missing
        field discovered during verification.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations: list[str] = violations
        summary = f"{len(violations)} violation(s): {violations[0]!r}" + (
            f" (and {len(violations) - 1} more)" if len(violations) > 1 else ""
        )
        super().__init__(f"Audit log integrity check failed — {summary}")


def _sha256_of_entry(entry: dict[str, Any]) -> str:
    """Return the hex SHA-256 digest of the canonical JSON form of *entry*.

    Canonical form: keys sorted, no extra whitespace, UTF-8 encoded.
    This is deterministic regardless of Python dict insertion order.
    """
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLogger:
    """Append-only, hash-chained audit logger.

    Each event written to ``log_path`` is a JSON object on its own line.
    The ``previous_hash`` field links each entry to its predecessor, forming
    an immutable chain that detects deletion, insertion, or modification of
    any entry.

    Parameters
    ----------
    log_path:
        Path to the NDJSON audit log file.  Parent directories are created
        on first write if they do not exist.
    tenant_id:
        Optional default tenant ID (unused internally; reserved for
        subclass convenience).

    Example
    -------
    >>> logger = AuditLogger("/var/log/pulsenet/audit.jsonl")
    >>> event_id = logger.log_event(
    ...     event_type="prediction_request",
    ...     tenant_id="acme-corp",
    ...     details={"model": "isolation_forest", "rul_estimate": 42},
    ... )
    >>> is_valid, violations = AuditLogger.verify_audit_log(
    ...     "/var/log/pulsenet/audit.jsonl"
    ... )
    """

    def __init__(
        self,
        log_path: str | os.PathLike[str],
        tenant_id: str = "",
    ) -> None:
        self.log_path = Path(log_path)
        self.default_tenant_id = tenant_id
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ────────────────────────────────────────────────────

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from disk.  Returns empty list if file absent."""
        if not self.log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _last_entry_hash(self) -> str:
        """Return SHA-256 of the last committed entry, or the genesis sentinel."""
        entries = self._read_entries()
        if not entries:
            return _GENESIS_HASH
        return _sha256_of_entry(entries[-1])

    # ── Public API ─────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        tenant_id: str,
        details: dict[str, Any],
    ) -> str:
        """Append a new event to the audit log and return its ``event_id``.

        Parameters
        ----------
        event_type:
            Short identifier for the event, e.g. ``"prediction_request"``,
            ``"rbac_violation"``, ``"adversarial_input_blocked"``.
        tenant_id:
            Identifier of the tenant that triggered the event.
        details:
            Arbitrary JSON-serialisable key/value context.

        Returns
        -------
        str
            UUID4 ``event_id`` assigned to the new entry.

        Raises
        ------
        ValueError
            If ``event_type`` or ``tenant_id`` are empty strings.
        TypeError
            If ``details`` is not a ``dict``.
        """
        if not event_type:
            raise ValueError("event_type must not be empty")
        if not tenant_id:
            raise ValueError("tenant_id must not be empty")
        if not isinstance(details, dict):
            raise TypeError(f"details must be a dict, got {type(details).__name__}")

        event_id = str(uuid.uuid4())
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        with _WRITE_LOCK:
            previous_hash = self._last_entry_hash()

            entry: dict[str, Any] = {
                "event_id": event_id,
                "timestamp": timestamp,
                "event_type": event_type,
                "tenant_id": tenant_id,
                "details": details,
                "previous_hash": previous_hash,
            }

            line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        return event_id

    # ── Static verification ────────────────────────────────────────────────

    @staticmethod
    def verify_audit_log(
        log_path: str | os.PathLike[str],
    ) -> tuple[bool, list[str]]:
        """Verify the hash-chain integrity of an audit log file.

        Replays every entry, recomputing the expected ``previous_hash`` for
        each link and comparing it to the stored value.  Also checks for
        required fields and non-decreasing timestamps.

        Parameters
        ----------
        log_path:
            Path to the NDJSON audit log file to verify.

        Returns
        -------
        tuple[bool, list[str]]
            ``(True, [])`` when the log is intact.  This method **never**
            returns ``(False, violations)`` — any integrity failure raises
            :exc:`AuditIntegrityError` immediately instead of allowing the
            caller to silently ignore the return value.

        Raises
        ------
        AuditIntegrityError
            If any hash-chain link is broken, any entry is missing required
            fields, any timestamp is non-monotonic, the file cannot be read,
            or the file does not exist.  The ``violations`` attribute of the
            exception contains the full list of human-readable descriptions.

        Example
        -------
        >>> try:
        ...     AuditLogger.verify_audit_log("audit.jsonl")
        ... except AuditIntegrityError as exc:
        ...     for v in exc.violations:
        ...         print("VIOLATION:", v)
        """
        path = Path(log_path)
        violations: list[str] = []

        if not path.exists():
            violations.append(f"Log file not found: {path}")
            raise AuditIntegrityError(violations)

        entries: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        violations.append(f"Line {lineno}: JSON parse error — {exc}")
        except OSError as exc:
            violations.append(f"Cannot read log file: {exc}")
            raise AuditIntegrityError(violations) from exc

        if violations:
            # JSON parse errors already found — fail immediately.
            raise AuditIntegrityError(violations)

        if not entries:
            return True, []  # Empty log is trivially valid.

        required_fields = {
            "event_id",
            "timestamp",
            "event_type",
            "tenant_id",
            "details",
            "previous_hash",
        }

        expected_previous_hash = _GENESIS_HASH
        previous_timestamp: str | None = None

        for idx, entry in enumerate(entries):
            # Field presence ---------------------------------------------------
            missing = required_fields - set(entry.keys())
            if missing:
                violations.append(
                    f"Entry {idx}: missing required fields: {sorted(missing)}"
                )
                # Advance expected hash to keep chain position accurate.
                expected_previous_hash = _sha256_of_entry(entry)
                continue

            # Hash chain -------------------------------------------------------
            stored = entry["previous_hash"]
            if stored != expected_previous_hash:
                violations.append(
                    f"Entry {idx} (event_id={entry.get('event_id', '?')}): "
                    f"hash chain broken — "
                    f"expected previous_hash={expected_previous_hash[:16]}…, "
                    f"got {stored[:16]}…"
                )

            # Timestamp monotonicity -------------------------------------------
            ts = entry.get("timestamp", "")
            if previous_timestamp is not None and ts < previous_timestamp:
                violations.append(
                    f"Entry {idx} (event_id={entry.get('event_id', '?')}): "
                    f"timestamp {ts!r} is earlier than previous "
                    f"entry {previous_timestamp!r}"
                )

            expected_previous_hash = _sha256_of_entry(entry)
            previous_timestamp = ts

        if violations:
            raise AuditIntegrityError(violations)

        return True, []
