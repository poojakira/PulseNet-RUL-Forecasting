"""
Sensor-spoofing attack (black-box, physically-realizable drift/spike patterns).

Threat model: Attacker has NO model access. Can only inject sensor readings
that stay within existing validation thresholds but corrupt downstream prediction.

Physically-realizable patterns based on real sensor failure modes:
- Gradual drift: sensor calibration drift over time (common in industrial sensors)
- Step change: sudden offset from loose wiring or electromagnetic interference
- Spike/glitch: transient spikes from electrical noise or grounding issues
- Stuck-at: sensor frozen at constant value (mechanical failure)
- Noise injection: increased variance from failing electronics

These patterns are designed to:
1. Stay under existing validation thresholds (range checks, rate-of-change limits)
2. Corrupt the feature engineering (rolling means, normalization)
3. Maximize impact on anomaly score while minimizing detectability
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from pulsenet.models.base import BaseAnomalyModel
from pulsenet.pipeline.preprocessing import (
    compute_rolling_features,
    get_feature_columns,
)


class SpoofingPattern(Enum):
    """Physically-realizable sensor spoofing patterns."""

    GRADUAL_DRIFT = "gradual_drift"  # Linear calibration drift
    STEP_CHANGE = "step_change"  # Sudden offset (loose connection)
    SPIKE_BURST = "spike_burst"  # Transient spikes (EMI/noise)
    STUCK_AT = "stuck_at"  # Frozen sensor value
    NOISE_INJECTION = "noise_injection"  # Increased variance
    CORRELATION_BREAK = "correlation_break"  # Break inter-sensor correlations


@dataclass(frozen=True)
class SpoofingResult:
    """Result of a sensor-spoofing attack trial."""

    success: bool
    pattern: SpoofingPattern
    original_score: float
    perturbed_score: float
    sensor_modified: str
    max_deviation: float  # Max absolute deviation from original
    validation_passed: bool  # Whether perturbed data passes validation
    detection_score: float  # Anomaly score on perturbed data


class SensorSpoofingAttack:
    """
    Black-box sensor spoofing attack with physically-realizable patterns.

    Attacker injects perturbed sensor readings that:
    - Pass existing range/rate-of-change validation
    - Corrupt downstream rolling features and normalization
    - Flip anomaly decision without triggering input validation
    """

    def __init__(
        self,
        model: BaseAnomalyModel,
        sensor_cols: list[str],
        validation_ranges: dict[str, tuple[float, float]] | None = None,
        max_rate_of_change: dict[str, float] | None = None,
    ):
        """
        Args:
            model: Target anomaly model
            sensor_cols: List of sensor column names
            validation_ranges: Dict of {sensor: (min, max)} for range validation
            max_rate_of_change: Dict of {sensor: max_delta_per_timestep}
        """
        self.model = model
        self.sensor_cols = sensor_cols
        self.validation_ranges = validation_ranges or {}
        self.max_rate_of_change = max_rate_of_change or {}

    def _apply_pattern(
        self,
        df: pd.DataFrame,
        pattern: SpoofingPattern,
        sensor: str,
        intensity: float = 1.0,
    ) -> pd.DataFrame:
        """Apply a spoofing pattern to a specific sensor column."""
        df = df.copy()
        original = df[sensor].values.copy()
        n = len(original)

        if pattern == SpoofingPattern.GRADUAL_DRIFT:
            # Linear drift: sensor slowly loses calibration
            drift = np.linspace(0, intensity * 2.0, n)  # Up to 2 sigma drift
            df[sensor] = original + drift

        elif pattern == SpoofingPattern.STEP_CHANGE:
            # Sudden offset at random point
            step_point = np.random.randint(n // 4, 3 * n // 4)
            offset = intensity * 3.0  # 3 sigma step
            df[sensor].iloc[step_point:] += offset

        elif pattern == SpoofingPattern.SPIKE_BURST:
            # Transient spikes at random positions
            n_spikes = max(1, int(n * 0.02 * intensity))  # 2% of points
            spike_indices = np.random.choice(n, n_spikes, replace=False)
            spike_magnitude = intensity * 5.0  # 5 sigma spikes
            df[sensor].iloc[spike_indices] += (
                np.random.choice([-1, 1], n_spikes) * spike_magnitude
            )

        elif pattern == SpoofingPattern.STUCK_AT:
            # Sensor frozen at a value (stuck at mean or extreme)
            stuck_value = np.mean(original) + intensity * np.std(original)
            df[sensor] = stuck_value

        elif pattern == SpoofingPattern.NOISE_INJECTION:
            # Increased variance from failing electronics
            noise = np.random.normal(0, intensity * np.std(original), n)
            df[sensor] = original + noise

        elif pattern == SpoofingPattern.CORRELATION_BREAK:
            # Break correlation with other sensors by adding uncorrelated noise
            noise = np.random.normal(0, intensity * np.std(original), n)
            df[sensor] = original + noise * 2.0  # Stronger to break correlation

        return df

    def _validate(self, df: pd.DataFrame) -> bool:
        """Check if perturbed data passes input validation."""
        for sensor, (min_val, max_val) in self.validation_ranges.items():
            if sensor in df.columns:
                vals = df[sensor].values
                if np.any(vals < min_val) or np.any(vals > max_val):
                    return False

        for sensor, max_roc in self.max_rate_of_change.items():
            if sensor in df.columns:
                diff = np.abs(np.diff(df[sensor].values))
                if np.any(diff > max_roc):
                    return False

        return True

    def attack(
        self,
        df: pd.DataFrame,
        pattern: SpoofingPattern,
        sensor: str,
        intensity: float = 1.0,
    ) -> SpoofingResult:
        """
        Run spoofing attack on a single unit's telemetry.

        Args:
            df: Sensor dataframe for one unit (must have unit_number, time_in_cycles)
            pattern: Spoofing pattern to apply
            sensor: Target sensor column
            intensity: Attack intensity multiplier

        Returns:
            SpoofingResult with success status and metrics
        """
        # Get original anomaly score (after preprocessing)
        original_score = self._get_score(df)

        # Apply spoofing pattern
        perturbed_df = self._apply_pattern(df, pattern, sensor, intensity)

        # Check validation
        validation_passed = self._validate(perturbed_df)

        # Get perturbed anomaly score
        perturbed_score = self._get_score(perturbed_df)

        # Max deviation from original
        max_dev = float(np.max(np.abs(perturbed_df[sensor].values - df[sensor].values)))

        # Success: validation passes AND anomaly decision flips
        original_pred = 1 if original_score > 0.5 else 0
        perturbed_pred = 1 if perturbed_score > 0.5 else 0
        success = validation_passed and (original_pred != perturbed_pred)

        return SpoofingResult(
            success=success,
            pattern=pattern,
            original_score=original_score,
            perturbed_score=perturbed_score,
            sensor_modified=sensor,
            max_deviation=max_dev,
            validation_passed=validation_passed,
            detection_score=perturbed_score,
        )

    def _get_score(self, df: pd.DataFrame) -> float:
        """Get anomaly score for a unit's telemetry after full preprocessing."""
        try:
            # Apply same preprocessing as pipeline
            feat_df = compute_rolling_features(df.copy())
            # For scoring, we need normalized features
            # Since we don't have train scaler here, use model's score on raw features
            feature_cols = get_feature_columns(feat_df)
            X = feat_df[feature_cols].values

            # Get score from model (handles its own preprocessing internally if needed)
            scores = self.model.score(X)
            return float(np.mean(scores))  # Average over timesteps
        except Exception:
            return 0.0

    def find_best_attack(
        self,
        df: pd.DataFrame,
        patterns: list[SpoofingPattern] | None = None,
        sensors: list[str] | None = None,
        intensities: list[float] | None = None,
    ) -> SpoofingResult:
        """Search for best spoofing attack across patterns/sensors/intensities."""
        patterns = patterns or list(SpoofingPattern)
        sensors = sensors or self.sensor_cols
        intensities = intensities or [0.5, 1.0, 1.5, 2.0]

        best_result = None
        best_score_diff = -1

        for pattern in patterns:
            for sensor in sensors:
                for intensity in intensities:
                    result = self.attack(df, pattern, sensor, intensity)
                    if result.validation_passed:
                        score_diff = abs(result.perturbed_score - result.original_score)
                        if score_diff > best_score_diff:
                            best_score_diff = score_diff
                            best_result = result
                            if result.success:
                                return result  # Early exit on success

        return best_result or SpoofingResult(
            success=False,
            pattern=patterns[0],
            original_score=0,
            perturbed_score=0,
            sensor_modified=sensors[0],
            max_deviation=0,
            validation_passed=False,
            detection_score=0,
        )
