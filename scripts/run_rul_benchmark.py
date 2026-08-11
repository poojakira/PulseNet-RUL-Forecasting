#!/usr/bin/env python3
"""Run an archived RUL regression benchmark on NASA C-MAPSS FD001-FD004.

Produces RMSE and the NASA C-MAPSS asymmetric scoring value for every subset,
using the official per-unit (chronological) train/test split. Emits a JSON file
and a RESULTS.md report to the requested local output directory.

Usage:
    python scripts/run_rul_benchmark.py --output-dir <dir> [--data-dir data/official]

Fails closed: if the verified NASA archive is not present it aborts. Results are
experimental simulator-data measurements, not real-world fleet evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pulsenet.evaluation.rul_regression import evaluate_subset  # noqa: E402
from pulsenet.pipeline.official_cmapss import (  # noqa: E402
    NASA_CMAPSS_LANDING_PAGE,
    NASA_CMAPSS_SHA256,
    NASA_CMAPSS_URL,
    load_official_subset,
)

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def main() -> int:
    parser = argparse.ArgumentParser(description="PulseNet RUL benchmark (C-MAPSS)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/official")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    archive = data_dir / "CMAPSSData.zip"
    if not archive.exists():
        raise SystemExit(
            f"ABORT: verified NASA C-MAPSS archive not found at {archive}. "
            f"Download it from {NASA_CMAPSS_URL} (SHA-256 {NASA_CMAPSS_SHA256}). "
            "Refusing to synthesize data."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    results = []
    archive_sha = None
    for subset in SUBSETS:
        data = load_official_subset(subset, data_dir)
        archive_sha = data.archive_sha256
        t0 = time.perf_counter()
        r = evaluate_subset(data.train, data.test, data.rul, subset=subset)
        elapsed = time.perf_counter() - t0
        row = r.to_dict()
        row["fit_eval_seconds"] = round(elapsed, 2)
        results.append(row)
        print(
            f"{subset}: RMSE={row['rmse']}  C-MAPSS score={row['cmapss_score']}  "
            f"({row['n_test_engines']} test engines, {elapsed:.1f}s)"
        )

    payload = {
        "generated_utc": now.isoformat(),
        "readiness": "local experimental result; not production evidence",
        "task": "NASA C-MAPSS Remaining-Useful-Life regression",
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": 200,
            "random_state": 42,
            "rul_target": "piecewise-linear, capped at 125 cycles",
            "features": "non-constant sensors + operating settings + per-unit rolling means (window=5)",
        },
        "split_methodology": (
            "Official NASA C-MAPSS per-unit split (chronological): the model is "
            "fit on the complete run-to-failure trajectories in train_FD00x.txt and "
            "evaluated at the last observed cycle of each disjoint engine in "
            "test_FD00x.txt against RUL_FD00x.txt. No random train/test split is "
            "used; no future cycle of a test engine is seen during training."
        ),
        "metrics": {
            "rmse": "root mean squared error at last observed cycle (cycles)",
            "cmapss_score": (
                "NASA asymmetric score sum: exp(-d/13)-1 for early (d<0), "
                "exp(d/10)-1 for late (d>=0), d = pred - true"
            ),
        },
        "dataset": {
            "name": "NASA C-MAPSS Turbofan Degradation Simulation",
            "source_url": NASA_CMAPSS_URL,
            "landing_page": NASA_CMAPSS_LANDING_PAGE,
            "archive_sha256": archive_sha,
            "archive_bytes": archive.stat().st_size,
            "retrieved_utc": now.isoformat(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "results": results,
    }

    json_path = output_dir / "rul_benchmark_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON -> {json_path}")

    md = _render_markdown(payload)
    md_path = output_dir / "RESULTS.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Saved report -> {md_path}")
    return 0


def _render_markdown(p: dict) -> str:
    ds = p["dataset"]
    lines = [
        "# PulseNet RUL Forecasting - Archived Local Benchmark",
        "",
        f"- Generated (UTC): {p['generated_utc']}",
        f"- Platform: {p['environment']['platform']} / Python {p['environment']['python']}",
        f"- Readiness: {p['readiness']}",
        "",
        "## Result summary",
        "",
        "| Subset | RMSE (cycles) | C-MAPSS score | Test engines |",
        "|--------|--------------:|--------------:|-------------:|",
    ]
    for r in p["results"]:
        lines.append(
            f"| {r['subset']} | {r['rmse']} | {r['cmapss_score']} | "
            f"{r['n_test_engines']} |"
        )
    lines += [
        "",
        "## Exact split methodology",
        "",
        p["split_methodology"],
        "",
        "For each training engine at cycle `t` with final cycle `T`, the RUL "
        "target is `min(T - t, 125)` (piecewise-linear RUL, the standard C-MAPSS "
        "convention). The scaler and regressor are fit on training engines only; "
        "prediction is made at the last observed cycle of each test engine and "
        "compared to `RUL_FD00x.txt`. Training and test engine sets are disjoint.",
        "",
        "## Model",
        "",
        f"- Type: {p['model']['type']} (n_estimators={p['model']['n_estimators']}, "
        f"random_state={p['model']['random_state']})",
        f"- Target: {p['model']['rul_target']}",
        f"- Features: {p['model']['features']}",
        "",
        "## Metrics defined",
        "",
        f"- RMSE: {p['metrics']['rmse']}",
        f"- C-MAPSS score: {p['metrics']['cmapss_score']}",
        "",
        "## Dataset provenance",
        "",
        f"- Name: {ds['name']}",
        f"- Source URL: {ds['source_url']}",
        f"- Landing page: {ds['landing_page']}",
        f"- SHA-256: `{ds['archive_sha256']}`",
        f"- Size: {ds['archive_bytes']} bytes",
        f"- Retrieved (UTC): {ds['retrieved_utc']}",
        "",
        "## Honest gaps and caveats",
        "",
        "- This is a classical experimental baseline on simulator data, not a "
        "field-validated maintenance model.",
        "- No comparison to published papers is made here because preprocessing, "
        "target construction, split semantics, and scoring must be reconciled "
        "before any external benchmark comparison is valid.",
        "- No per-operating-condition normalization is applied; FD002/FD004 "
        "would benefit from it. The RandomForest partially compensates by "
        "splitting on the operating-setting features.",
        "- RUL is capped at 125 for both training targets and reported metrics, "
        "following common C-MAPSS practice; absolute numbers depend on this cap.",
        "- These numbers must not be published as product performance without a "
        "new, independently reviewed evaluation protocol.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
