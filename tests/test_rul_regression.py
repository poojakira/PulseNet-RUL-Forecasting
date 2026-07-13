"""Tests for the RUL regression module (RMSE + C-MAPSS asymmetric score).

Covers the scoring maths and the official per-unit split evaluation. The
integration test uses the real NASA C-MAPSS archive if it is present under
``data/official`` and is skipped otherwise (no synthetic data is fabricated).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pulsenet.evaluation.rul_regression import (
    cmapss_score,
    evaluate_subset,
    piecewise_linear_rul,
    rmse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = PROJECT_ROOT / "data" / "official" / "CMAPSSData.zip"


class TestScoringFunctions:
    def test_perfect_prediction_is_zero(self):
        y = np.array([10.0, 50.0, 125.0])
        assert cmapss_score(y, y) == pytest.approx(0.0)
        assert rmse(y, y) == pytest.approx(0.0)

    def test_cmapss_late_penalised_more_than_early(self):
        # Same absolute error of 10 cycles.
        late = cmapss_score(np.array([50.0]), np.array([60.0]))  # d = +10
        early = cmapss_score(np.array([50.0]), np.array([40.0]))  # d = -10
        assert late == pytest.approx(math.exp(10 / 10) - 1)
        assert early == pytest.approx(math.exp(10 / 13) - 1)
        assert late > early  # asymmetry: over-prediction hurts more

    def test_cmapss_is_additive_over_units(self):
        y_true = np.array([50.0, 50.0])
        y_pred = np.array([60.0, 40.0])
        expected = (math.exp(10 / 10) - 1) + (math.exp(10 / 13) - 1)
        assert cmapss_score(y_true, y_pred) == pytest.approx(expected)

    def test_rmse_value(self):
        assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
            math.sqrt((9 + 16) / 2)
        )

    def test_piecewise_linear_rul_caps(self):
        rul = piecewise_linear_rul(np.array([0, 100, 190]), max_cycle=200, rul_cap=125)
        # 200-0=200 -> capped to 125; 200-100=100 -> 100; 200-190=10 -> 10
        np.testing.assert_allclose(rul, [125.0, 100.0, 10.0])


@pytest.mark.skipif(not _ARCHIVE.exists(), reason="NASA C-MAPSS archive not present")
class TestOfficialSplitEvaluation:
    def test_fd001_uses_official_per_unit_split(self):
        from pulsenet.pipeline.official_cmapss import load_official_subset

        data = load_official_subset("FD001", PROJECT_ROOT / "data" / "official")
        # Official split: train and test engine-id sets are disjoint trajectories,
        # and there is one ground-truth RUL per test engine.
        assert data.rul.shape[0] == data.test["unit_number"].nunique()

        result = evaluate_subset(
            data.train, data.test, data.rul, subset="FD001", n_estimators=50
        )
        assert result.n_test_engines == 100
        assert "no random split" in result.split
        # Honest sanity band: a classical baseline lands in the published
        # ~15-25 RMSE range on FD001; assert it is finite and reasonable.
        assert 10.0 < result.rmse < 30.0
        assert result.cmapss_score > 0.0
