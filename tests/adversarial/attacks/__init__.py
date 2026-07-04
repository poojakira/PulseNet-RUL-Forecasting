"""
Adversarial Telemetry Attack Suite for PulseNet.

Three attack vectors against sensor input space:
1. Gradient-based perturbation (white-box PGD-style for regression/time-series)
2. Sensor-spoofing (black-box, physically-realizable drift/spike patterns)
3. Replay/staleness (valid-but-stale telemetry windows)
"""

from __future__ import annotations

from .gradient_perturbation import GradientPerturbationAttack, PGDAttackResult
from .sensor_spoofing import SensorSpoofingAttack, SpoofingResult
from .replay_staleness import ReplayStalenessAttack, ReplayResult

__all__ = [
    "GradientPerturbationAttack",
    "PGDAttackResult",
    "SensorSpoofingAttack",
    "SpoofingResult",
    "ReplayStalenessAttack",
    "ReplayResult",
]