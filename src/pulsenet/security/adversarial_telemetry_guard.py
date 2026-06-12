"""
adversarial_telemetry_guard.py - Adversarial telemetry detection for safety-critical ML
Author: Pooja Kiran (github.com/poojakira)

Detects sensor-level attacks on ML inference input in safety-critical contexts.
Reference: f7i.ai 2026 (sensor masking attacks), MITRE ATT&CK for ICS T0839

Gap: Safety-critical ML inference (turbofan RUL prediction) is vulnerable to
sensor-level manipulation. A compromised sensor reading fed to a pre-trained
model can cause catastrophic RUL under-prediction or over-prediction.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TelemetryAlertCode(Enum):
    REPLAY = "REPLAY_ATTACK"
    FROZEN_SENSOR = "FROZEN_SENSOR"
    SUDDEN_JUMP = "SUDDEN_JUMP"
    SENSOR_BIAS = "SENSOR_BIAS_INJECTION"


@dataclass
class TelemetryAlert:
    code: TelemetryAlertCode
    severity: str  # CRITICAL | HIGH | MEDIUM
    sensor_index: Optional[int]
    detail: str


@dataclass
class TelemetryGuardConfig:
    frozen_consecutive_threshold: int = 5
    max_single_step_delta: float = 0.5
    bias_sigma_threshold: float = 3.0
    baseline_window: int = 50


class AdversarialTelemetryGuard:
    """
    Inline adversarial telemetry detection for NASA C-MAPSS turbofan RUL.

    Checks every telemetry frame before it reaches the RUL model.
    Detects: replay attacks, frozen sensors, sudden jumps, sensor bias injection.

    NOTE: Uses z-score for bias detection (assumes approximate stationarity
    within baseline window). Non-stationary degradation over long sequences
    requires CUSUM or ADWIN - planned for future version.

    Usage:
        guard = AdversarialTelemetryGuard(n_sensors=21)
        alerts = guard.check(telemetry_frame)
        if alerts:
            # reject frame or flag for human review
    """

    def __init__(self, n_sensors: int, config: Optional[TelemetryGuardConfig] = None):
        self.n_sensors = n_sensors
        self.cfg = config or TelemetryGuardConfig()
        self._history: deque[list[float]] = deque(maxlen=self.cfg.baseline_window)
        self._consecutive_counts: list[int] = [0] * n_sensors
        self._prev_frame: Optional[list[float]] = None
        self._seen_hashes: set[str] = set()

    def _frame_hash(self, frame: list[float]) -> str:
        return hashlib.sha256(json.dumps(frame, sort_keys=True).encode()).hexdigest()

    def check(self, frame: list[float]) -> list[TelemetryAlert]:
        """
        Check a telemetry frame for adversarial patterns.

        Args:
            frame: list of sensor readings (length must equal n_sensors)

        Returns:
            List of TelemetryAlert (empty = clean)
        """
        if len(frame) != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} sensors, got {len(frame)}")

        alerts: list[TelemetryAlert] = []

        # --- Replay attack: exact frame seen before ---
        fhash = self._frame_hash(frame)
        if fhash in self._seen_hashes:
            alerts.append(
                TelemetryAlert(
                    code=TelemetryAlertCode.REPLAY,
                    severity="HIGH",
                    sensor_index=None,
                    detail=f"Exact frame replay detected. hash={fhash[:16]}. MITRE ICS T0839.",
                )
            )
        self._seen_hashes.add(fhash)

        # --- Frozen sensor: same value for N consecutive frames ---
        if self._prev_frame is not None:
            for i, (prev_val, curr_val) in enumerate(zip(self._prev_frame, frame)):
                if abs(curr_val - prev_val) < 1e-9:
                    self._consecutive_counts[i] += 1
                    if (
                        self._consecutive_counts[i]
                        >= self.cfg.frozen_consecutive_threshold
                    ):
                        alerts.append(
                            TelemetryAlert(
                                code=TelemetryAlertCode.FROZEN_SENSOR,
                                severity="HIGH",
                                sensor_index=i,
                                detail=(
                                    f"Sensor {i} frozen for "
                                    f"{self._consecutive_counts[i]} consecutive frames. "
                                    "Possible sensor masking attack."
                                ),
                            )
                        )
                else:
                    self._consecutive_counts[i] = 0

        # --- Sudden jump: single-step delta exceeds physical plausibility ---
        if self._prev_frame is not None:
            for i, (prev_val, curr_val) in enumerate(zip(self._prev_frame, frame)):
                delta = abs(curr_val - prev_val)
                if delta > self.cfg.max_single_step_delta:
                    alerts.append(
                        TelemetryAlert(
                            code=TelemetryAlertCode.SUDDEN_JUMP,
                            severity="HIGH",
                            sensor_index=i,
                            detail=(
                                f"Sensor {i} sudden jump: delta={delta:.4f} > "
                                f"threshold={self.cfg.max_single_step_delta}. "
                                "Possible bias injection attack."
                            ),
                        )
                    )

        # --- Sensor bias injection: z-score vs rolling baseline ---
        if len(self._history) >= self.cfg.baseline_window // 2:
            history_list = list(self._history)
            for i in range(self.n_sensors):
                sensor_baseline = [h[i] for h in history_list]
                try:
                    mean = statistics.mean(sensor_baseline)
                    stdev = statistics.stdev(sensor_baseline)
                    if stdev > 0:
                        z = abs(frame[i] - mean) / stdev
                        if z > self.cfg.bias_sigma_threshold:
                            alerts.append(
                                TelemetryAlert(
                                    code=TelemetryAlertCode.SENSOR_BIAS,
                                    severity="MEDIUM",
                                    sensor_index=i,
                                    detail=(
                                        f"Sensor {i} z-score={z:.2f} > "
                                        f"threshold={self.cfg.bias_sigma_threshold}. "
                                        "Possible slow bias injection."
                                    ),
                                )
                            )
                except statistics.StatisticsError:
                    pass

        self._prev_frame = list(frame)
        self._history.append(list(frame))
        return alerts
