"""
Tests for RUL forecasting and maintenance scheduling.

These are fast unit tests that use a small synthetic fixture for code-path
correctness. They are not product performance evidence. The deep-model benchmark
is an archived local script in benchmark/deep_rul_benchmark.py.
"""

import numpy as np
import pandas as pd
import pytest

from pulsenet.models.rul_forecaster import (
    MaintenanceAction,
    MaintenanceScheduler,
    RULForecast,
    RULForecaster,
    cmapss_score,
    piecewise_linear_rul,
    rmse,
)


def _synthetic_run_to_failure(n_engines: int = 6, length: int = 60) -> pd.DataFrame:
    """Build a small synthetic run-to-failure frame with a degrading sensor."""
    rows = []
    rng = np.random.default_rng(0)
    for unit in range(1, n_engines + 1):
        for t in range(1, length + 1):
            frac = t / length
            rows.append(
                {
                    "unit_number": unit,
                    "time_in_cycles": t,
                    "op_setting_1": 0.0,
                    "sensor_1": 1.0,  # constant (should be dropped)
                    "sensor_2": 100.0 - 20.0 * frac + rng.normal(0, 0.5),  # degrades
                    "sensor_3": 50.0 + 10.0 * frac + rng.normal(0, 0.5),  # rises
                }
            )
    return pd.DataFrame(rows)


class TestScoringFunctions:
    def test_rmse_zero_when_exact(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y) == 0.0

    def test_cmapss_penalizes_late_more_than_early(self):
        y_true = np.array([50.0])
        early = cmapss_score(y_true, np.array([40.0]))  # predicted less life
        late = cmapss_score(y_true, np.array([60.0]))  # predicted more life
        assert late > early, "late predictions must be penalized more"

    def test_piecewise_linear_caps(self):
        rul = piecewise_linear_rul(np.array([0, 10, 200]), max_cycle=200, cap=125)
        assert rul.max() <= 125
        assert rul[0] == 125  # 200 - 0 capped at 125


class TestRULForecaster:
    def test_fit_predict_shapes_and_bounds(self):
        train = _synthetic_run_to_failure()
        # Test engines truncated before failure
        test = _synthetic_run_to_failure(n_engines=3, length=30)
        f = RULForecaster(window=5, n_estimators=50).fit(train)
        forecasts = f.predict_last_cycle(test)
        assert len(forecasts) == 3
        for fc in forecasts:
            assert isinstance(fc, RULForecast)
            assert 0.0 <= fc.predicted_rul <= 125.0
            assert fc.lower_rul <= fc.predicted_rul  # conservative <= point estimate

    def test_predict_before_fit_raises(self):
        f = RULForecaster()
        with pytest.raises(RuntimeError, match="fit"):
            f.predict_last_cycle(_synthetic_run_to_failure())

    def test_learns_degradation_direction(self):
        """An engine early in life should get a higher RUL than one near failure."""
        train = _synthetic_run_to_failure(n_engines=8, length=80)
        f = RULForecaster(window=5, n_estimators=80).fit(train)
        young = _synthetic_run_to_failure(n_engines=1, length=10)  # lots of life left
        old = _synthetic_run_to_failure(n_engines=1, length=75)  # near failure
        rul_young = f.predict_last_cycle(young)[0].predicted_rul
        rul_old = f.predict_last_cycle(old)[0].predicted_rul
        assert rul_young > rul_old, f"young={rul_young} should exceed old={rul_old}"


class TestMaintenanceScheduler:
    def test_thresholds_must_increase(self):
        with pytest.raises(ValueError):
            MaintenanceScheduler(
                immediate_threshold=50, plan_threshold=40, monitor_threshold=80
            )

    def test_immediate_action_when_rul_low(self):
        s = MaintenanceScheduler()
        fc = RULForecast(
            unit_number=1, predicted_rul=12.0, lower_rul=8.0, confidence_margin=4.0
        )
        d = s.decide(fc)
        assert d.action == MaintenanceAction.IMMEDIATE

    def test_plan_action_mid_range(self):
        s = MaintenanceScheduler()
        fc = RULForecast(
            unit_number=2, predicted_rul=35.0, lower_rul=30.0, confidence_margin=5.0
        )
        d = s.decide(fc)
        assert d.action == MaintenanceAction.PLAN

    def test_healthy_when_rul_high(self):
        s = MaintenanceScheduler()
        fc = RULForecast(
            unit_number=3, predicted_rul=120.0, lower_rul=110.0, confidence_margin=10.0
        )
        d = s.decide(fc)
        assert d.action == MaintenanceAction.HEALTHY

    def test_decision_acts_on_conservative_rul(self):
        """Point estimate healthy but conservative estimate triggers PLAN  --  safety first."""
        s = MaintenanceScheduler(plan_threshold=40)
        fc = RULForecast(
            unit_number=4, predicted_rul=55.0, lower_rul=38.0, confidence_margin=17.0
        )
        d = s.decide(fc)
        assert d.action == MaintenanceAction.PLAN, (
            "must use conservative RUL, not point estimate"
        )

    def test_decision_serializes(self):
        s = MaintenanceScheduler()
        fc = RULForecast(
            unit_number=5, predicted_rul=12.0, lower_rul=8.0, confidence_margin=4.0
        )
        d = s.decide(fc).to_dict()
        assert d["action"] == "immediate"
        assert d["unit_number"] == 5
