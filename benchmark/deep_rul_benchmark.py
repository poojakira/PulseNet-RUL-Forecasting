"""
Archived experimental RUL benchmark: 1D-CNN on NASA C-MAPSS FD001.

Trains the deep sequence model on the NASA simulator dataset and writes a local
JSON result file. The output is not release evidence and must not be compared to
published papers unless preprocessing, target construction, split, and scoring
are independently reconciled.

Run:
    python benchmark/deep_rul_benchmark.py --output /tmp/pulsenet-fd001.json
"""

from __future__ import annotations

import hashlib
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


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name, sep=r"\s+", header=None, engine="python")
    df.columns = COLS[: df.shape[1]]
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local JSON output path. Use /tmp or another non-committed location.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    train = _load("train_FD001.txt")
    test = _load("test_FD001.txt")
    rul = pd.read_csv(DATA / "RUL_FD001.txt", header=None)[0].to_numpy(dtype=float)

    t0 = time.perf_counter()
    res = train_and_evaluate(train, test, rul, epochs=args.epochs, seed=args.seed)
    elapsed = round(time.perf_counter() - t0, 1)

    print(f"1D-CNN RUL  RMSE = {res.rmse}  |  C-MAPSS score = {res.cmapss_score}")
    print(f"Trained in {elapsed}s on CPU.")

    evidence = {
        "schema_version": "deep-rul-fd001-local-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "readiness": "local experimental result; not production evidence",
        "dataset": "NASA C-MAPSS FD001 simulator data",
        "data_source": "data/official/CMAPSSData",
        "data_sha256": {
            "train_FD001.txt": _sha256(DATA / "train_FD001.txt"),
            "test_FD001.txt": _sha256(DATA / "test_FD001.txt"),
            "RUL_FD001.txt": _sha256(DATA / "RUL_FD001.txt"),
        },
        "model": res.model,
        "window": res.window,
        "epochs": res.epochs,
        "seed": args.seed,
        "rmse": res.rmse,
        "cmapss_score": res.cmapss_score,
        "n_test_engines": res.n_test_engines,
        "train_seconds_cpu": elapsed,
        "methodology": (
            "Sliding 30-cycle windows over 14 informative sensors, min-max scaled "
            "on train only. Piecewise-linear RUL target capped at 125. Evaluated on "
            "the last window of each test engine vs RUL_FD001. No random split, no "
            "published-baseline comparison is made by this script."
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
