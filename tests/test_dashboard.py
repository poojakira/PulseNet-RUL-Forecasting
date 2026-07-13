"""Smoke/integration test for the Streamlit dashboard via AppTest.

Runs the real dashboard script end-to-end against a prepared working directory
(synthetic feature CSV + trained Isolation Forest model), exercising the data
loading, ML scoring, RUL estimation, and rendering code paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsenet.models.isolation_forest import IsolationForestModel

DASHBOARD = (
    Path(__file__).resolve().parents[1] / "src" / "pulsenet" / "dashboard" / "app.py"
)


@pytest.fixture
def dashboard_workspace(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for unit in (1, 2, 3):
        for cycle in range(1, 40):
            row = {"unit_number": unit, "time_in_cycles": cycle}
            for s in range(2, 10):
                row[f"sensor_{s}"] = float(rng.random())
            rows.append(row)
    df = pd.DataFrame(rows)

    feature_cols = [
        c
        for c in df.columns
        if c not in ("unit_number", "time_in_cycles", "is_anomaly")
    ]
    model = IsolationForestModel(n_estimators=20)
    model.train(df[feature_cols].to_numpy())

    (tmp_path / "models").mkdir()
    model.save(tmp_path / "models" / "isolation_forest.joblib")
    df.to_csv(tmp_path / "test_features.csv", index=False)

    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_dashboard_runs(dashboard_workspace):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()

    assert not at.exception, f"Dashboard raised: {at.exception}"
    # Title should reflect the selected engine unit
    assert any("Engine Unit" in str(t.value) for t in at.title)
