"""
Generate real dashboard data from the RUL forecaster on NASA C-MAPSS FD001.

Runs the actual GradientBoosting RUL forecaster on the committed real dataset,
produces a per-engine RUL prediction and maintenance decision, and writes
dashboard/data.json. The HTML dashboard fetches this file - no hardcoded
numbers, no Math.random. Re-run this to refresh the dashboard from real data.

Run:
    python dashboard/generate_dashboard_data.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pulsenet.models.rul_forecaster import (  # noqa: E402
    MaintenanceScheduler,
    RULForecaster,
    rmse,
)

DATA = REPO / "data" / "official" / "CMAPSSData"
COLS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name, sep=r"\s+", header=None, engine="python")
    df.columns = COLS[: df.shape[1]]
    return df


def main() -> int:
    train = _load("train_FD001.txt")
    test = _load("test_FD001.txt")
    rul_truth = pd.read_csv(DATA / "RUL_FD001.txt", header=None)[0].to_numpy(dtype=float)

    forecaster = RULForecaster().fit(train)
    scheduler = MaintenanceScheduler()
    forecasts = forecaster.predict_last_cycle(test)

    # Real accuracy on this run (predicted vs ground-truth, capped at 125).
    y_pred = [fc.predicted_rul for fc in forecasts]
    y_true = [min(v, 125.0) for v in rul_truth]
    fleet_rmse = round(rmse(y_true, [min(p, 125.0) for p in y_pred]), 2)

    engines = []
    for fc in forecasts:
        d = scheduler.decide(fc)
        engines.append(
            {
                "unit": fc.unit_number,
                "predicted_rul": fc.predicted_rul,
                "conservative_rul": fc.lower_rul,
                "action": d.action.value,
                "reason": d.reason,
            }
        )

    # Sort most-urgent first (lowest conservative RUL).
    engines.sort(key=lambda e: e["conservative_rul"])
    action_counts = Counter(e["action"] for e in engines)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "NASA C-MAPSS FD001 (real, committed)",
        "model": "GradientBoosting RUL forecaster",
        "fleet_size": len(engines),
        "fleet_rmse_cycles": fleet_rmse,
        "action_summary": {
            "immediate": action_counts.get("immediate", 0),
            "plan": action_counts.get("plan", 0),
            "monitor": action_counts.get("monitor", 0),
            "healthy": action_counts.get("healthy", 0),
        },
        "engines": engines,
    }

    out = REPO / "dashboard" / "data.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Fleet: {len(engines)} engines, RMSE {fleet_rmse} cycles")
    print(f"Actions: {dict(action_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
