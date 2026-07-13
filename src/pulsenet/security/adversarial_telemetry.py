"""Adversarial telemetry guardrails for inference-time robustness checks."""

from __future__ import annotations

import os
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
        enable_perturbation: bool | None = None,
    ):
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.ood_z_threshold = ood_z_threshold
        self.perturb_eps = perturb_eps
        self.finite_diff_step = finite_diff_step
        self.max_rows_for_perturbation = max_rows_for_perturbation
        # Finite-difference gradient estimation costs 2 model calls per feature
        # per row, i.e. it multiplies inference cost dramatically. It can be
        # disabled to bound worst-case compute (denial-of-wallet defense).
        # Precedence: explicit arg > PULSENET_ADV_TELEMETRY_PERTURB env > on.
        if enable_perturbation is None:
            env_flag = os.environ.get("PULSENET_ADV_TELEMETRY_PERTURB", "1")
            enable_perturbation = env_flag.lower() not in ("0", "false", "no")
        self.enable_perturbation = enable_perturbation

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

    @staticmethod
    def _is_torch_module(model: Any) -> bool:
        """True if *model* is a torch.nn.Module (without importing torch when
        it is not installed)."""
        try:
            import torch  # noqa: F401

            import torch.nn as nn

            return isinstance(model, nn.Module)
        except Exception:
            return False

    def _torch_gradient(self, model: Any, row: np.ndarray) -> np.ndarray:
        """Exact gradient via autograd for a PyTorch model.

        This is O(1) backward passes vs O(n_features) forward passes for finite
        differences. We use the model output's scalar sum as the score proxy.
        """
        import torch

        x = torch.tensor(
            row.reshape(1, -1), dtype=torch.float32, requires_grad=True
        )
        out = model(x)
        score = out.sum()
        (grad,) = torch.autograd.grad(score, x)
        return grad.detach().numpy().reshape(-1)

    def _approximate_gradient(self, model: Any, row: np.ndarray) -> np.ndarray:
        # Prefer exact autograd for torch models. Finite differences below are
        # the fallback for the project's BaseAnomalyModel contract, which
        # exposes ``score(X_2d) -> per-row scores`` (sklearn-style). PyTorch
        # ``nn.Module`` objects have NO ``.score`` method, so calling it here
        # would raise/mislead — we route them to autograd instead.
        if self._is_torch_module(model):
            return self._torch_gradient(model, row)
        if not hasattr(model, "score"):
            raise TypeError(
                "AdversarialTelemetryGuard requires either a torch.nn.Module or "
                "a model exposing score(X_2d) -> per-row scores; got "
                f"{type(model).__name__} with neither."
            )
        grad = np.zeros_like(row, dtype=float)
        for idx in range(row.shape[0]):
            delta = np.zeros_like(row, dtype=float)
            delta[idx] = self.finite_diff_step
            plus = float(model.score((row + delta).reshape(1, -1))[0])
            minus = float(model.score((row - delta).reshape(1, -1))[0])
            grad[idx] = (plus - minus) / (2.0 * self.finite_diff_step)
        return grad

    def _score_rows(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return per-row scores for either a torch model or a .score() model."""
        if self._is_torch_module(model):
            import torch

            with torch.no_grad():
                out = model(torch.tensor(X, dtype=torch.float32))
                if out.ndim > 1:
                    out = out.sum(dim=1)
            return out.detach().numpy().reshape(-1)
        return np.asarray(model.score(X), dtype=float)

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
        deltas: list[float] = []
        if self.enable_perturbation:
            base_scores = self._score_rows(model, arr[:sampled])
            for row_idx in range(sampled):
                grad = self._approximate_gradient(model, arr[row_idx])
                perturbed = arr[row_idx] + self.perturb_eps * np.sign(grad)
                perturbed_score = float(
                    self._score_rows(model, perturbed.reshape(1, -1))[0]
                )
                deltas.append(abs(perturbed_score - float(base_scores[row_idx])))
        else:
            sampled = 0

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
