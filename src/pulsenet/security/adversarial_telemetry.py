"""Adversarial telemetry guardrails for inference-time robustness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TelemetryGuardResult:
    """Structured output for perturbation and OOD telemetry checks."""

    ood_detected: bool
    ood_fraction: float
    max_ood_zscore: float
    max_perturbation_delta: float
    mean_perturbation_delta: float
    sampled_rows: int


class AdversarialTelemetryGuard:
    """Finite-difference perturbation simulation + simple OOD z-score checks."""

    def __init__(
        self,
        reference_mean: np.ndarray | None = None,
        reference_std: np.ndarray | None = None,
        ood_z_threshold: float = 4.0,
        perturb_eps: float = 0.01,
        finite_diff_step: float = 1e-3,
        max_rows_for_perturbation: int = 4,
    ):
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.ood_z_threshold = ood_z_threshold
        self.perturb_eps = perturb_eps
        self.finite_diff_step = finite_diff_step
        self.max_rows_for_perturbation = max_rows_for_perturbation

    @classmethod
    def from_scaler(cls, scaler: Any) -> "AdversarialTelemetryGuard":
        """Bootstrap OOD references from MinMax-like scaler bounds."""
        if scaler is None:
            return cls()

        data_min = getattr(scaler, "data_min_", None)
        data_max = getattr(scaler, "data_max_", None)
        if data_min is None or data_max is None:
            return cls()

        data_min_arr = np.asarray(data_min, dtype=float)
        data_max_arr = np.asarray(data_max, dtype=float)
        midpoint = (data_min_arr + data_max_arr) / 2.0
        spread = np.maximum((data_max_arr - data_min_arr) / 2.0, 1e-6)
        return cls(reference_mean=midpoint, reference_std=spread)

    def fit_reference(self, X_ref: np.ndarray) -> None:
        """Fit OOD reference distribution from trusted baseline telemetry."""
        arr = np.asarray(X_ref, dtype=float)
        self.reference_mean = arr.mean(axis=0)
        self.reference_std = np.maximum(arr.std(axis=0), 1e-6)

    def _approximate_gradient(self, model: Any, row: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(row, dtype=float)
        for idx in range(row.shape[0]):
            delta = np.zeros_like(row, dtype=float)
            delta[idx] = self.finite_diff_step
            plus = float(model.score((row + delta).reshape(1, -1))[0])
            minus = float(model.score((row - delta).reshape(1, -1))[0])
            grad[idx] = (plus - minus) / (2.0 * self.finite_diff_step)
        return grad

    def _ood_stats(self, X: np.ndarray) -> tuple[bool, float, float]:
        if self.reference_mean is None or self.reference_std is None:
            return False, 0.0, 0.0

        z = np.abs((X - self.reference_mean) / np.maximum(self.reference_std, 1e-6))
        row_max = z.max(axis=1)
        flags = row_max >= self.ood_z_threshold
        return bool(np.any(flags)), float(np.mean(flags)), float(np.max(row_max))

    def assess(self, model: Any, X: np.ndarray) -> TelemetryGuardResult:
        """Run OOD check plus adversarial perturbation simulation."""
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return TelemetryGuardResult(False, 0.0, 0.0, 0.0, 0.0, 0)

        sampled = min(arr.shape[0], self.max_rows_for_perturbation)
        base_scores = np.asarray(model.score(arr[:sampled]), dtype=float)
        deltas: list[float] = []
        for row_idx in range(sampled):
            grad = self._approximate_gradient(model, arr[row_idx])
            perturbed = arr[row_idx] + self.perturb_eps * np.sign(grad)
            perturbed_score = float(model.score(perturbed.reshape(1, -1))[0])
            deltas.append(abs(perturbed_score - float(base_scores[row_idx])))

        ood_detected, ood_fraction, max_ood_z = self._ood_stats(arr)
        max_delta = max(deltas) if deltas else 0.0
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        return TelemetryGuardResult(
            ood_detected=ood_detected,
            ood_fraction=ood_fraction,
            max_ood_zscore=max_ood_z,
            max_perturbation_delta=max_delta,
            mean_perturbation_delta=mean_delta,
            sampled_rows=sampled,
        )
