"""
Structured logging module for PulseNet.

Usage:
    from pulsenet.logger import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started", extra={"phase": "ingestion", "records": 500})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge extra fields (skip internal logging keys)
        skip = {
            "name",
            "msg",
            "args",
            "created",
            "relativeCreated",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "filename",
            "module",
            "pathname",
            "thread",
            "threadName",
            "processName",
            "process",
            "levelname",
            "levelno",
            "message",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in skip:
                payload[k] = v
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable coloured log lines for local dev."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"{color}{ts} [{record.levelname:<8}]{self.RESET} {record.name}: {record.getMessage()}"


def get_logger(name: str, level: str = "INFO", fmt: str = "text") -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR).
    fmt : str
        ``"json"`` for structured output, ``"text"`` for dev.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter() if fmt == "json" else _TextFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
