"""
External log shipping for tamper-evident audit trails.

Solves Finding 3: the local hash-chained log is rewritable by anyone with
disk access. This module ships log entries to external WORM (Write Once Read
Many) destinations where an attacker cannot modify history.

Supported backends:
- AWS CloudWatch Logs (with retention lock)
- AWS S3 with Object Lock (WORM compliance mode)
- Stdout (for container deployments where the orchestrator captures logs)

In production, configure at least ONE external destination. The local
hash-chain remains as a fast verification mechanism, but the external
copy is the source of truth for forensics.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol


class LogShipper(Protocol):
    """Interface for external log destinations."""

    def ship(self, entry: dict[str, Any]) -> bool:
        """Ship a log entry. Returns True on success."""
        ...


class CloudWatchShipper:
    """Ship logs to AWS CloudWatch Logs.

    The log group should have a retention policy and be in a separate
    AWS account from the application (security account pattern).

    Requires: pip install boto3
    Set: PULSENET_CW_LOG_GROUP, PULSENET_CW_LOG_STREAM
    """

    def __init__(self) -> None:
        import boto3

        self._client = boto3.client("logs")
        self._log_group = os.environ.get(
            "PULSENET_CW_LOG_GROUP", "/pulsenet/security-audit"
        )
        self._log_stream = os.environ.get("PULSENET_CW_LOG_STREAM", "audit-trail")
        self._sequence_token: str | None = None

    def ship(self, entry: dict[str, Any]) -> bool:
        """Ship one entry to CloudWatch Logs."""
        try:
            kwargs: dict[str, Any] = {
                "logGroupName": self._log_group,
                "logStreamName": self._log_stream,
                "logEvents": [
                    {
                        "timestamp": int(time.time() * 1000),
                        "message": json.dumps(entry, default=str),
                    }
                ],
            }
            if self._sequence_token:
                kwargs["sequenceToken"] = self._sequence_token
            response = self._client.put_log_events(**kwargs)
            self._sequence_token = response.get("nextSequenceToken")
            return True
        except Exception:
            return False


class S3WORMShipper:
    """Ship logs to S3 with Object Lock (WORM).

    The bucket MUST have Object Lock enabled with COMPLIANCE mode.
    Once written, neither the application nor an attacker can delete
    or modify the log entry until the retention period expires.

    Requires: pip install boto3
    Set: PULSENET_S3_AUDIT_BUCKET
    """

    def __init__(self) -> None:
        import boto3

        self._client = boto3.client("s3")
        self._bucket = os.environ.get("PULSENET_S3_AUDIT_BUCKET", "pulsenet-audit-worm")

    def ship(self, entry: dict[str, Any]) -> bool:
        """Ship one entry as an S3 object with Object Lock retention."""
        try:
            timestamp = entry.get("timestamp", time.time())
            key = f"audit/{int(timestamp)}/{entry.get('event_type', 'unknown')}.json"
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(entry, default=str).encode(),
                ContentType="application/json",
                ObjectLockMode="COMPLIANCE",
                ObjectLockRetainUntilDate=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(time.time() + 365 * 86400),  # 1 year retention
                ),
            )
            return True
        except Exception:
            return False


class StdoutShipper:
    """Ship logs to stdout in JSON format.

    For container deployments (ECS, K8s) where the orchestrator captures
    stdout and ships to a centralized log system (CloudWatch Container
    Insights, Datadog, Splunk).
    """

    def ship(self, entry: dict[str, Any]) -> bool:
        """Print structured JSON to stdout."""
        print(
            json.dumps({"_source": "pulsenet-audit", **entry}, default=str), flush=True
        )
        return True


def create_shippers() -> list[LogShipper]:
    """Create configured log shippers based on environment variables.

    Returns at minimum the stdout shipper. Add CloudWatch/S3 based on config.
    """
    shippers: list[LogShipper] = [StdoutShipper()]

    if os.environ.get("PULSENET_CW_LOG_GROUP"):
        try:
            shippers.append(CloudWatchShipper())
        except ImportError:
            pass

    if os.environ.get("PULSENET_S3_AUDIT_BUCKET"):
        try:
            shippers.append(S3WORMShipper())
        except ImportError:
            pass

    return shippers
