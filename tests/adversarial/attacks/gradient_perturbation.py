"""
Gradient-based perturbation attack (white-box PGD-style for regression/time-series).

Adaptation of Projected Gradient Descent (PGD) for anomaly detection regression:
- Standard PGD: minimizes classification loss to flip discrete labels
- This adaptation: maximizes anomaly score (regression output) to cross detection threshold
- For Isolation Forest: maximizes decision_function (more negative = more anomalous)
- For LSTM: maximizes reconstruction error
- For Ensemble: maximizes weighted anomaly score

Citation: Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
Adaptation for regression/time-series anomaly detection per:
  - "Adversarial Attacks on Time Series" (Zhao et al., 2019)
  - "Adversarial Examples for Anomaly Detection" (Kravchik & Shabtai, 2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pulsenet.models.base import BaseAnomalyModel
from pulsenet.models.ensemble import EnsembleModel
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.models.lstm_model import LSTMModel


@dataclass(frozen=True)
class PGDAttackResult:
    """Result of a gradient-based perturbation attack trial."""

    success: bool
    original_score: float
    perturbed_score: float
    perturbation_magnitude: float  # L2 norm of perturbation
    perturbation_linf: float  # L-infinity norm (max per-feature change)
    iterations: int
    model_name: str


class GradientPerturbationAttack:
    """
    White-box PGD-style attack on sensor inputs to flip anomaly decision.

    Threat model: Attacker has full model access (architecture, weights, preprocessing).
    Goal: Minimal sensor-value perturbations that flip RUL prediction or anomaly-gate decision.
    """

    def __init__(
        self,
        model: BaseAnomalyModel,
        epsilon: float = 0.1,
        step_size: float = 0.01,
        max_iter: int = 100,
        norm: str = "l2",
        target_score: Optional[float] = None,
    ):
        """
        Args:
            model: Target anomaly model (must expose score/decision_function)
            epsilon: Maximum perturbation budget (L2 or Linf norm)
            step_size: PGD step size
            max_iter: Maximum PGD iterations
            norm: "l2" or "linf" perturbation constraint
            target_score: Target anomaly score to exceed (if None, uses model's threshold)
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_iter = max_iter
        self.norm = norm
        self.target_score = target_score

    def _get_gradient(self, X: np.ndarray, target: float = 1.0) -> np.ndarray:
        """
        Compute gradient of anomaly score w.r.t. input.
        Uses finite differences since models may not be fully differentiable (IsolationForest).
        """
        eps = 1e-4
        grad = np.zeros_like(X)
        base_score = self.model.score(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += eps
            score_plus = self.model.score(X_plus)
            grad[:, i] = (score_plus - base_score) / eps

        return grad

    def _project(self, delta: np.ndarray, epsilon: float) -> np.ndarray:
        """Project perturbation onto L2 or Linf ball."""
        if self.norm == "l2":
            norms = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, epsilon / (norms + 1e-12))
            return delta * scale
        else:  # linf
            return np.clip(delta, -epsilon, epsilon)

    def attack(self, X: np.ndarray) -> PGDAttackResult:
        """
        Run PGD attack on a single sample (or batch).
        Returns result for the first sample if batch provided.
        """
        X = np.atleast_2d(X)
        original_X = X.copy()

        # Get original score
        original_score = float(self.model.score(original_X)[0])

        # Determine target: push score above threshold
        if self.target_score is not None:
            target = self.target_score
        elif hasattr(self.model, "threshold") and self.model.threshold is not None:
            target = float(self.model.threshold) * 1.1  # exceed threshold by 10%
        else:
            # For Isolation Forest: push decision_function more negative
            target = (
                original_score * 2.0 if original_score > 0 else original_score * 0.5
            )

        delta = np.zeros_like(X)

        for iteration in range(self.max_iter):
            # Compute gradient of score w.r.t. input
            grad = self._get_gradient(X + delta)

            # PGD step: move in direction that increases anomaly score
            if self.norm == "l2":
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                step = self.step_size * grad / (grad_norm + 1e-12)
            else:
                step = self.step_size * np.sign(grad)

            delta += step
            delta = self._project(delta, self.epsilon)

            # Check if attack succeeded
            perturbed_X = original_X + delta
            perturbed_score = float(self.model.score(perturbed_X)[0])

            if perturbed_score >= target:
                return PGDAttackResult(
                    success=True,
                    original_score=original_score,
                    perturbed_score=perturbed_score,
                    perturbation_magnitude=float(np.linalg.norm(delta[0])),
                    perturbation_linf=float(np.max(np.abs(delta[0]))),
                    iterations=iteration + 1,
                    model_name=self.model.name,
                )

        # Attack failed
        perturbed_score = float(self.model.score(original_X + delta)[0])
        return PGDAttackResult(
            success=False,
            original_score=original_score,
            perturbed_score=perturbed_score,
            perturbation_magnitude=float(np.linalg.norm(delta[0])),
            perturbation_linf=float(np.max(np.abs(delta[0]))),
            iterations=self.max_iter,
            model_name=self.model.name,
        )

    def attack_batch(self, X: np.ndarray) -> list[PGDAttackResult]:
        """Run attack on each sample in batch."""
        return [self.attack(X[i : i + 1]) for i in range(len(X))]


def load_trained_ensemble(model_dir: Path) -> EnsembleModel:
    """Load trained ensemble from disk."""
    ensemble = EnsembleModel()
    ensemble.load(model_dir)
    return ensemble


def load_trained_isolation_forest(model_path: Path) -> IsolationForestModel:
    """Load trained Isolation Forest from disk."""
    model = IsolationForestModel()
    model.load(model_path, trusted=True)
    return model


def load_trained_lstm(model_path: Path) -> LSTMModel:
    """Load trained LSTM from disk."""
    model = LSTMModel()
    model.load(model_path)
    return model
