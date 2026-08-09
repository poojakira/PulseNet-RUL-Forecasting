"""
Production RUL benchmark: 1D-CNN on NASA C-MAPSS FD001.

Trains the deep sequence model on the official per-unit split and writes a
committed evidence file with RMSE + NASA C-MAPSS score, alongside published
baselines for context. Every number here is measured on this machine from the
real dataset shipped in data/official/CMAPSSData - no synthetic values.

Run:
    python benchmark/deep_rul_benchmark.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pulsenet.models.deep_rul_cnn import train_and_evaluate  # noqa: E402

DATA = REPO / "data" / "official" / "CMAPSSData"
COLS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Published FD001 RMSE, for honest context (not our numbers).
PUBLISHED = {
    "RandomForest (classical, this repo)": 18.25,
    "CNN - Babu et al. 2016": 18.45,
    "LSTM - Zheng et al. 2017": 16.14,
    "DCNN - Li et al. 2018": 12.61,
}


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name, sep=r"\s+", header=None, engine="python")
    df.columns = COLS[: df.shape[1]]
    return df


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "evidence" / "deep_rul_fd001.json",
    )
    args = parser.parse_args()

    train = _load("train_FD001.txt")
    test = _load("test_FD001.txt")
    rul = pd.read_csv(DATA / "RUL_FD001.txt", header=None)[0].to_numpy(dtype=float)

    t0 = time.perf_counter()
    res = train_and_evaluate(train, test, rul, epochs=args.epochs, seed=args.seed)
    elapsed = round(time.perf_counter() - t0, 1)

    beats = [name for name, val in PUBLISHED.items() if res.rmse < val]

    print(f"1D-CNN RUL  RMSE = {res.rmse}  |  C-MAPSS score = {res.cmapss_score}")
    print(f"Trained in {elapsed}s on CPU. Beats: {', '.join(beats)}")

    evidence = {
        "schema_version": "deep-rul-fd001-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "NASA C-MAPSS FD001 (official per-unit train/test split)",
        "data_source": "data/official/CMAPSSData (committed, real)",
        "model": res.model,
        "window": res.window,
        "epochs": res.epochs,
        "seed": args.seed,
        "rmse": res.rmse,
        "cmapss_score": res.cmapss_score,
        "n_test_engines": res.n_test_engines,
        "train_seconds_cpu": elapsed,
        "published_baselines_rmse": PUBLISHED,
        "beats_baselines": beats,
        "methodology": (
            "Sliding 30-cycle windows over 14 informative sensors, min-max scaled "
            "on train only. Piecewise-linear RUL target capped at 125. Evaluated on "
            "the last window of each test engine vs RUL_FD001. No random split, no "
            "temporal leakage. All values measured on this machine."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Evidence written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
