"""
Production RUL forecasting and maintenance scheduling for NASA C-MAPSS.

This is the core product capability: given live sensor telemetry from a turbofan
engine, predict Remaining Useful Life (RUL) in cycles and convert that into an
actionable maintenance decision.

Two layers:

1. ``RULForecaster`` - a gradient-boosted regressor over engineered degradation
   features (rolling mean/std/slope per sensor). Trained and evaluated on the
   official C-MAPSS per-unit split with the standard piecewise-linear RUL target
   and the NASA asymmetric scoring function. No random split, no temporal leakage.

2. ``MaintenanceScheduler`` - turns an RUL prediction (plus its uncertainty) into
   one of four operational actions with a lead-time buffer, so a planner can act
   on the forecast instead of just reading a number.

Design choices grounded in the maintenance-scheduling reality:
- Late predictions (saying an engine has more life than it does) are dangerous,
  so the scheduler subtracts a safety margin derived from model residual spread
  before deciding. This mirrors the asymmetric C-MAPSS cost.
- Actions map to real planning horizons (immediate / plan / monitor / healthy),
  not just a raw number, because that is what a maintenance team consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

_META_COLS = ("unit_number", "time_in_cycles")


# ─── Scoring ───────────────────────────────────────────────────────────────────


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric score (lower is better).

    Late predictions (pred > true) penalised on exp(d/10); early on exp(-d/13).
    Reference: Saxena et al., PHM 2008.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(
        np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0))
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


# ─── Feature engineering ─────────────────────────────────────────────────────


def _informative_sensors(train: pd.DataFrame, threshold: float = 1e-6) -> list[str]:
    """Keep sensor/op columns that actually vary (drop constant channels)."""
    candidate = [c for c in train.columns if c not in _META_COLS]
    var = train[candidate].var(axis=0, numeric_only=True)
    return [c for c in candidate if float(var.get(c, 0.0)) > threshold]


def _engineer(df: pd.DataFrame, sensors: list[str], window: int) -> pd.DataFrame:
    """Add causal rolling mean, std, and slope per engine unit.

    All windows are backward-looking (only past/current cycles), so features are
    valid for a truncated test trajectory - the exact online scenario.
    """
    out = df.copy()
    g = out.groupby("unit_number")
    for col in sensors:
        out[f"{col}_rmean"] = g[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out[f"{col}_rstd"] = g[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0.0)
        )
        # Slope: difference over the window, a simple degradation-rate proxy.
        out[f"{col}_slope"] = g[col].transform(
            lambda s: s.diff(periods=window).fillna(0.0)
        )
    return out


def _feature_columns(sensors: list[str]) -> list[str]:
    cols: list[str] = list(sensors)
    for c in sensors:
        cols += [f"{c}_rmean", f"{c}_rstd", f"{c}_slope"]
    return cols


def piecewise_linear_rul(
    cycles: np.ndarray, max_cycle: float, cap: float
) -> np.ndarray:
    """min(max_cycle - cycle, cap) - the standard C-MAPSS RUL target."""
    return np.minimum(max_cycle - np.asarray(cycles, float), cap)


# ─── Forecaster ──────────────────────────────────────────────────────────────


@dataclass
class RULForecast:
    """One engine's RUL prediction with an uncertainty band."""

    unit_number: int
    predicted_rul: float
    lower_rul: float  # conservative estimate (predicted - safety margin)
    confidence_margin: float


class RULForecaster:
    """Gradient-boosted RUL regressor over engineered degradation features."""

    def __init__(
        self,
        rul_cap: int = 125,
        window: int = 10,
        random_state: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
    ) -> None:
        self.rul_cap = rul_cap
        self.window = window
        self._scaler = StandardScaler()
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            random_state=random_state,
        )
        self._sensors: list[str] = []
        self._feature_cols: list[str] = []
        self._residual_std: float = 0.0
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> RULForecaster:
        train = train.copy()
        self._sensors = _informative_sensors(train)
        train = _engineer(train, self._sensors, self.window)
        self._feature_cols = _feature_columns(self._sensors)

        y_parts: list[np.ndarray] = []
        for _, unit in train.groupby("unit_number"):
            y_parts.append(
                piecewise_linear_rul(
                    unit["time_in_cycles"].to_numpy(),
                    float(unit["time_in_cycles"].max()),
                    self.rul_cap,
                )
            )
        y_train = np.concatenate(y_parts)

        x_train = self._scaler.fit_transform(train[self._feature_cols].to_numpy())
        self._model.fit(x_train, y_train)

        # Residual spread on training data drives the safety margin used by
        # the scheduler. Computed once, honestly, from in-sample residuals.
        resid = y_train - self._model.predict(x_train)
        self._residual_std = float(np.std(resid))
        self._fitted = True
        return self

    def predict_last_cycle(self, test: pd.DataFrame) -> list[RULForecast]:
        """Predict RUL at the last observed cycle of each test engine."""
        if not self._fitted:
            raise RuntimeError("call fit() first")
        test = _engineer(test.copy(), self._sensors, self.window)
        last = (
            test.sort_values(["unit_number", "time_in_cycles"])
            .groupby("unit_number")
            .tail(1)
            .sort_values("unit_number")
        )
        x = self._scaler.transform(last[self._feature_cols].to_numpy())
        preds = np.clip(self._model.predict(x), 0.0, self.rul_cap)
        margin = 1.28 * self._residual_std  # ~90% one-sided coverage
        return [
            RULForecast(
                unit_number=int(u),
                predicted_rul=round(float(p), 2),
                lower_rul=round(float(max(0.0, p - margin)), 2),
                confidence_margin=round(margin, 2),
            )
            for u, p in zip(last["unit_number"].to_numpy(), preds, strict=True)
        ]


# ─── Maintenance scheduling ──────────────────────────────────────────────────


class MaintenanceAction(str, Enum):
    IMMEDIATE = "immediate"  # ground the asset now
    PLAN = "plan"  # schedule within the planning window
    MONITOR = "monitor"  # increased inspection cadence
    HEALTHY = "healthy"  # normal operation


@dataclass
class MaintenanceDecision:
    unit_number: int
    predicted_rul: float
    conservative_rul: float
    action: MaintenanceAction
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class MaintenanceScheduler:
    """Convert an RUL forecast into an operational maintenance action.

    Thresholds are expressed in cycles and applied to the CONSERVATIVE RUL
    (prediction minus the model's uncertainty margin), so the decision is
    robust to the model over-estimating remaining life - the dangerous
    direction in predictive maintenance.
    """

    def __init__(
        self,
        immediate_threshold: int = 15,
        plan_threshold: int = 40,
        monitor_threshold: int = 80,
    ) -> None:
        if not (immediate_threshold < plan_threshold < monitor_threshold):
            raise ValueError("thresholds must be strictly increasing")
        self.immediate = immediate_threshold
        self.plan = plan_threshold
        self.monitor = monitor_threshold

    def decide(self, forecast: RULForecast) -> MaintenanceDecision:
        rul = forecast.lower_rul  # act on the conservative estimate
        if rul <= self.immediate:
            action = MaintenanceAction.IMMEDIATE
            reason = (
                f"Conservative RUL {rul:.0f} cycles <= {self.immediate}. "
                "Ground the asset; failure risk within the safety buffer."
            )
        elif rul <= self.plan:
            action = MaintenanceAction.PLAN
            reason = (
                f"Conservative RUL {rul:.0f} cycles <= {self.plan}. "
                "Schedule maintenance within the planning window."
            )
        elif rul <= self.monitor:
            action = MaintenanceAction.MONITOR
            reason = (
                f"Conservative RUL {rul:.0f} cycles <= {self.monitor}. "
                "Increase inspection cadence; degradation trend detectable."
            )
        else:
            action = MaintenanceAction.HEALTHY
            reason = (
                f"Conservative RUL {rul:.0f} cycles > {self.monitor}. Normal operation."
            )
        return MaintenanceDecision(
            unit_number=forecast.unit_number,
            predicted_rul=forecast.predicted_rul,
            conservative_rul=forecast.lower_rul,
            action=action,
            reason=reason,
        )
