#!/usr/bin/env python3
"""Replay official NASA FD001 telemetry into a running PulseNet API."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pulsenet.pipeline.official_cmapss import load_official_fd001  # noqa: E402

SENSOR_COLUMNS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FD001-REPLAY] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fd001-replay")


def authenticate(api_url: str, username: str, password: str) -> str:
    response = requests.post(
        f"{api_url}/token",
        json={"username": username, "password": password},
        timeout=10,
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    log.info("Authenticated replay user '%s'", username)
    return str(token)


def iter_official_frames(
    data_dir: Path,
    unit_number: int,
    limit: int,
) -> list[dict[str, Any]]:
    fd001 = load_official_fd001(data_dir, max_train_rows=None, max_test_rows=None)
    unit = fd001.test[fd001.test["unit_number"] == unit_number].head(limit)
    if unit.empty:
        raise ValueError(f"unit {unit_number} not present in FD001 test data")
    return [
        {col: float(row[col]) for col in SENSOR_COLUMNS} for _, row in unit.iterrows()
    ]


def replay(
    api_url: str,
    token: str,
    frames: list[dict[str, Any]],
    tenant_id: str,
    delay_seconds: float,
) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Tenant-ID": tenant_id,
    }
    for index, payload in enumerate(frames, start=1):
        start = time.perf_counter()
        response = requests.post(
            f"{api_url}/api/v1/predict",
            json=payload,
            headers=headers,
            timeout=15,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code != 200:
            log.warning(
                "frame=%s status=%s body=%s", index, response.status_code, response.text
            )
        else:
            body = response.json()
            log.info(
                "frame=%s latency_ms=%.1f status=%s health=%.2f",
                index,
                latency_ms,
                body.get("status"),
                body.get("health_index"),
            )
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url", default=os.getenv("PULSENET_API_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--username", default=os.getenv("PULSENET_REPLAY_USER", "operator")
    )
    parser.add_argument("--password-env", default="PULSENET_REPLAY_PASSWORD")
    parser.add_argument("--tenant-id", default="official-nasa")
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data/official"))
    args = parser.parse_args()

    password = os.getenv(args.password_env)
    if not password:
        raise SystemExit(f"{args.password_env} must be set")

    token = authenticate(args.api_url, args.username, password)
    frames = iter_official_frames(args.data_dir, args.unit, args.limit)
    replay(args.api_url, token, frames, args.tenant_id, args.delay_seconds)


if __name__ == "__main__":
    main()
