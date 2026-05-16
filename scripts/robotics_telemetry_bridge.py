#!/usr/bin/env python3
"""
PulseNet Edge Telemetry Bridge — Simulated IoT Client.

Streams synthetic sensor readings to a running PulseNet API. Useful for
demoing the API + dashboard without wiring up real hardware. The simulated
"engine" gradually drifts an HPC-temperature sensor upward to exercise the
anomaly path.

Configuration is read from environment variables (12-factor style):

    PULSENET_API_URL          (default: http://127.0.0.1:8000)
    PULSENET_API_USERNAME     (default: operator)
    PULSENET_API_PASSWORD     (no default — must be set)
    PULSENET_API_CYCLES       (default: 100)
    PULSENET_API_HZ           (default: 1.0)
    PULSENET_API_TIMEOUT_S    (default: 5.0)

Example:
    export PULSENET_API_PASSWORD=ops123
    python scripts/robotics_telemetry_bridge.py
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [edge] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telemetry-bridge")


def _auth_endpoint(api_url: str) -> str:
    return f"{api_url.rstrip('/')}/token"


def _predict_endpoint(api_url: str) -> str:
    return f"{api_url.rstrip('/')}/api/v1/predict"


def authenticate(api_url: str, username: str, password: str, timeout: float) -> str:
    log.info("Authenticating as '%s' against %s ...", username, api_url)
    try:
        r = requests.post(
            _auth_endpoint(api_url),
            json={"username": username, "password": password},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        log.error("Cannot connect to PulseNet API at %s (%s)", api_url, e)
        sys.exit(2)
    except requests.exceptions.Timeout:
        log.error("Auth request timed out after %ss", timeout)
        sys.exit(2)

    if r.status_code != 200:
        log.error("Auth failed: %d %s", r.status_code, r.text[:200])
        sys.exit(1)
    token = r.json().get("access_token")
    if not token:
        log.error("Auth response had no access_token: %s", r.text[:200])
        sys.exit(1)
    log.info("Authenticated. Token acquired.")
    return token


def run_telemetry_stream(
    api_url: str,
    token: str,
    cycles: int,
    hz: float,
    timeout: float,
    rng: random.Random,
) -> int:
    """Stream `cycles` readings at the requested rate. Returns last cycle number."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Baseline values from C-MAPSS FD001 documentation (healthy engine).
    baseline = {
        "sensor_2": 642.15, "sensor_3": 1589.0, "sensor_4": 1406.2,
        "sensor_7": 554.1, "sensor_8": 2388.0, "sensor_9": 9044.8,
        "sensor_11": 47.5, "sensor_12": 521.9, "sensor_13": 2388.1,
        "sensor_14": 8138.6, "sensor_15": 8.44, "sensor_17": 393.0,
        "sensor_20": 39.0, "sensor_21": 23.4,
    }
    period = 1.0 / hz if hz > 0 else 1.0
    drift = 0.0

    log.info("Streaming %d cycles at %.2f Hz to %s ...", cycles, hz, api_url)
    cycle = 0
    for cycle in range(1, cycles + 1):
        drift += rng.uniform(0.1, 0.5)
        payload = {k: v + (rng.uniform(-0.005, 0.005) * v) for k, v in baseline.items()}
        payload["sensor_4"] += drift * 2.0  # gradual HPC outlet temp rise

        t0 = time.perf_counter()
        try:
            r = requests.post(_predict_endpoint(api_url), json=payload, headers=headers, timeout=timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code == 200:
                d = r.json()
                health = float(d.get("health_index", 100.0))
                status = d.get("status", "UNKNOWN")
                log.info("cycle=%04d health=%.1f%% status=%s latency=%.1fms",
                         cycle, health, status, latency_ms)
                if health < 50.0:
                    log.error("CRITICAL: health=%.1f%% — would trigger maintenance alert", health)
                    break
            else:
                log.warning("API returned %d: %s", r.status_code, r.text[:200])
        except requests.exceptions.RequestException as e:
            log.error("Request failed: %s", e)
        time.sleep(period)
    return cycle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("PULSENET_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--username", default=os.environ.get("PULSENET_API_USERNAME", "operator"))
    parser.add_argument(
        "--password",
        default=os.environ.get("PULSENET_API_PASSWORD"),
        help="Password (or set PULSENET_API_PASSWORD)",
    )
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("PULSENET_API_CYCLES", "100")))
    parser.add_argument("--hz", type=float, default=float(os.environ.get("PULSENET_API_HZ", "1.0")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("PULSENET_API_TIMEOUT_S", "5.0")))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.password:
        log.error("Password not provided. Set PULSENET_API_PASSWORD or pass --password.")
        sys.exit(2)

    rng = random.Random(args.seed)
    token = authenticate(args.api_url, args.username, args.password, args.timeout)
    last_cycle = run_telemetry_stream(args.api_url, token, args.cycles, args.hz, args.timeout, rng)
    log.info("Done. Last cycle transmitted: %d", last_cycle)


if __name__ == "__main__":
    main()
