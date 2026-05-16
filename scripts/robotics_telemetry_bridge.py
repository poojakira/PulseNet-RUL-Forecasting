#!/usr/bin/env python3
"""
PulseNet Edge Telemetry Bridge — Simulated IoT Client

Simulates an edge controller deployed on a turbofan engine test stand.
Continuously transmits live sensor readings to the PulseNet API and receives
real-time health predictions. If AI scores health below the critical threshold
(50%), it triggers a safe-shutdown alert.

Usage:
    python scripts/robotics_telemetry_bridge.py

Requires:
    - PulseNet API running on localhost:8000
    - Valid operator credentials configured
"""

import json
import logging
import random
import sys
import time

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TelemetryBridge")

API_URL = "http://localhost:8000"
AUTH_ENDPOINT = f"{API_URL}/token"
PREDICT_ENDPOINT = f"{API_URL}/api/v1/predict"


def authenticate(username: str = "operator", password: str = "ops123") -> str:
    """Authenticate with PulseNet API and return JWT token."""
    log.info(f"Authenticating as '{username}'...")
    try:
        response = requests.post(
            AUTH_ENDPOINT, json={"username": username, "password": password}
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            log.info("Authentication successful. Token acquired.")
            return token
        else:
            log.error(f"Authentication failed: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to PulseNet API. Is the server running?")
        sys.exit(1)


def alert_critical_health(health_score: float) -> None:
    """Log critical health alert (in production, this would trigger PLC shutdown)."""
    log.error("=" * 60)
    log.error(f"CRITICAL: Health score {health_score:.1f}% below threshold")
    log.error("In production: would trigger maintenance alert / safe shutdown")
    log.error("=" * 60)


def run_telemetry_stream(token: str, cycles: int = 100) -> None:
    """
    Simulate sensor telemetry stream at 1Hz.

    Sensor values are based on typical C-MAPSS FD001 ranges for a healthy engine,
    with gradual degradation injected to simulate compressor wear.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Baseline sensor values (typical healthy C-MAPSS FD001 ranges)
    base_sensors = {
        "sensor_2": 642.15,
        "sensor_3": 1589.0,
        "sensor_4": 1406.2,
        "sensor_7": 554.1,
        "sensor_8": 2388.0,
        "sensor_9": 9044.8,
        "sensor_11": 47.5,
        "sensor_12": 521.9,
        "sensor_13": 2388.1,
        "sensor_14": 8138.6,
        "sensor_15": 8.44,
        "sensor_17": 393.0,
        "sensor_20": 39.0,
        "sensor_21": 23.4,
    }

    degradation_factor = 0.0
    log.info(f"Starting telemetry stream ({cycles} cycles at 1Hz)...")

    for cycle in range(1, cycles + 1):
        # Simulate gradual degradation (HPC temperature drift)
        degradation_factor += random.uniform(0.1, 0.5)

        # Add sensor noise + degradation
        payload = {
            k: v + (random.uniform(-0.005, 0.005) * v)
            for k, v in base_sensors.items()
        }
        # Specifically degrade HPC outlet temperature (sensor_4)
        payload["sensor_4"] += degradation_factor * 2.0

        t0 = time.time()
        try:
            res = requests.post(PREDICT_ENDPOINT, json=payload, headers=headers)
            latency_ms = (time.time() - t0) * 1000

            if res.status_code == 200:
                data = res.json()
                health = data.get("health_index", 100.0)
                status = data.get("status", "UNKNOWN")

                log.info(
                    f"Cycle {cycle:04d} | "
                    f"Health={health:.1f}% | "
                    f"Status={status} | "
                    f"Latency={latency_ms:.1f}ms"
                )

                if health < 50.0:
                    alert_critical_health(health)
                    break
            else:
                log.warning(f"API returned {res.status_code}: {res.text[:100]}")
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed: {e}")

        time.sleep(1.0)

    log.info(f"Telemetry stream complete. {cycle} cycles transmitted.")


if __name__ == "__main__":
    print("PulseNet Edge Telemetry Bridge v1.0")
    print("=" * 40)
    token = authenticate()
    run_telemetry_stream(token)
