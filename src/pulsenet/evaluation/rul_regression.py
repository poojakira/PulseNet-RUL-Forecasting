# pyright: reportGeneralTypeIssues=false
"""Remaining-Useful-Life (RUL) regression for NASA C-MAPSS.

This module provides the *actual* RUL forecasting capability implied by the
project name.  The existing pipeline only performed binary anomaly detection
(Isolation Forest) and never produced an RUL estimate, an RMSE, or the NASA
C-MAPSS asymmetric scoring value.  This module fills that gap with an honest,
classical regression baseline evaluated on the **official per-unit split**.

Split methodology (chronological / per-unit — NOT random)
---------------------------------------------------------
The C-MAPSS archive already ships a proper split by engine unit:

* ``train_FD00x.txt`` — complete run-to-failure trajectories for a set of
  engines.  For a training row at cycle ``t`` of an engine whose final cycle is
  ``T``, the RUL target is ``min(T - t, rul_cap)`` (piecewise-linear RUL, the
  standard C-MAPSS convention; degradation is assumed negligible early in life).
* ``test_FD00x.txt`` — trajectories for a **disjoint** set of engines, each
  truncated at some point *before* failure.
* ``RUL_FD00x.txt`` — the ground-truth RUL at the **last observed cycle** of
  each test engine.

We fit the scaler and the regressor on the training engines only, then predict
the RUL at the last observed cycle of every test engine and compare against the
ground truth.  Training and test engines are disjoint and no future cycle of a
test engine is ever seen during training, so there is no temporal leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Metadata columns that are never used as model features.
_META_COLS = ("unit_number", "time_in_cycles")


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric scoring function (lower is better).

    Late predictions (predicting more life than remains, ``d = pred - true > 0``)
    are penalised more heavily than early predictions, reflecting the higher
    operational cost of a missed failure.

        d_i = y_pred_i - y_true_i
        s_i = exp(-d_i / 13) - 1   if d_i <  0   (early / conservative)
        s_i = exp( d_i / 10) - 1   if d_i >= 0   (late / dangerous)
        score = sum_i s_i

    Reference: Saxena et al., "Damage Propagation Modeling for Aircraft Engine
    Run-to-Failure Simulation", PHM 2008.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def piecewise_linear_rul(
    cycles: np.ndarray, max_cycle: float, rul_cap: float
) -> np.ndarray:
    """Piecewise-linear RUL target: ``min(max_cycle - cycle, rul_cap)``."""
    rul = max_cycle - np.asarray(cycles, dtype=float)
    return np.minimum(rul, rul_cap)


def select_feature_columns(
    train: pd.DataFrame, variance_threshold: float = 1e-6
) -> list[str]:
    """Drop metadata and (near-)constant sensor/setting columns.

    Sensors that never move on a given subset carry no degradation signal and
    can destabilise scaling; this reproduces the standard C-MAPSS practice of
    dropping constant channels, chosen automatically per subset by variance.
    """
    candidate = [c for c in train.columns if c not in _META_COLS]
    # Wrap in pd.Series so static type checkers know ``.get`` is valid
    # (DataFrame.var(axis=0) returns a Series, but stubs widen it to float).
    variances = pd.Series(train[candidate].var(axis=0, numeric_only=True))
    return [
        c
        for c in candidate
        if c in variances.index and float(variances.loc[c]) > variance_threshold
    ]


def _add_rolling_features(
    df: pd.DataFrame, feature_cols: list[str], window: int
) -> pd.DataFrame:
    """Add per-unit rolling-mean features (causal: only past/current cycles)."""
    out = df.copy()
    grouped = out.groupby("unit_number")
    for col in feature_cols:
        out[f"{col}_rmean"] = grouped[col].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
    return out


@dataclass
class RulEvaluationResult:
    subset: str
    rmse: float
    cmapss_score: float
    n_test_engines: int
    n_train_engines: int
    n_train_rows: int
    rul_cap: int
    rolling_window: int
    n_features: int
    split: str = "official per-unit train/test (chronological, no random split)"

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_subset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul: pd.Series,
    *,
    subset: str = "FD00x",
    rul_cap: int = 125,
    rolling_window: int = 5,
    random_state: int = 42,
    n_estimators: int = 200,
) -> RulEvaluationResult:
    """Fit an RUL regressor on the training engines and evaluate on test engines.

    Returns RMSE and the C-MAPSS asymmetric score computed at the last observed
    cycle of each test engine (the canonical C-MAPSS evaluation protocol).
    """
    train = train.copy()
    test = test.copy()

    feature_cols = select_feature_columns(train)

    train = _add_rolling_features(train, feature_cols, rolling_window)
    test = _add_rolling_features(test, feature_cols, rolling_window)
    model_cols = feature_cols + [f"{c}_rmean" for c in feature_cols]

    # --- Build training targets: piecewise-linear RUL per training engine ---
    y_parts: list[np.ndarray] = []
    for _, unit_df in train.groupby("unit_number"):
        max_cycle = float(unit_df["time_in_cycles"].max())
        y_parts.append(
            piecewise_linear_rul(
                unit_df["time_in_cycles"].to_numpy(), max_cycle, rul_cap
            )
        )
    y_train = np.concatenate(y_parts)

    # --- Scale on train only (no leakage), transform test ---
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train[model_cols].to_numpy())

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    # --- Predict at the LAST observed cycle of each test engine ---
    last_rows = (
        test.sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number")
        .tail(1)
        .sort_values("unit_number")
    )
    x_test = scaler.transform(last_rows[model_cols].to_numpy())
    y_pred = np.clip(model.predict(x_test), 0.0, None)

    # Ground truth is ordered by ascending unit id in RUL_FD00x.txt.
    y_true = np.minimum(rul.to_numpy(dtype=float), rul_cap)
    y_pred_capped = np.minimum(y_pred, rul_cap)

    return RulEvaluationResult(
        subset=subset,
        rmse=round(rmse(y_true, y_pred_capped), 4),
        cmapss_score=round(cmapss_score(y_true, y_pred_capped), 4),
        n_test_engines=int(last_rows["unit_number"].nunique()),
        n_train_engines=int(train["unit_number"].nunique()),
        n_train_rows=int(len(train)),
        rul_cap=rul_cap,
        rolling_window=rolling_window,
        n_features=len(model_cols),
    )
