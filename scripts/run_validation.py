"""Produce validation metrics from official NASA C-MAPSS FD001 data."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from pulsenet.models.isolation_forest import IsolationForestModel  # noqa: E402
from pulsenet.pipeline.official_cmapss import load_official_fd001  # noqa: E402
from pulsenet.pipeline.preprocessing import (  # noqa: E402
    compute_rolling_features,
    create_labels,
    get_feature_columns,
    normalize,
)


def write_metric_chart(chart_metrics: dict[str, float], path: Path) -> None:
    """Write a small SVG chart from measured official-data metrics."""
    width = 720
    height = 260
    margin_left = 120
    bar_height = 28
    gap = 22
    scale_width = 500
    bars = []
    for idx, (name, value) in enumerate(chart_metrics.items()):
        y = 42 + idx * (bar_height + gap)
        bar_width = max(1, int(value * scale_width))
        bars.append(
            f'<text x="20" y="{y + 20}" font-size="14" font-family="Arial">{name}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{scale_width}" height="{bar_height}" '
            'fill="#e8edf3" />'
            f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" '
            'fill="#2563eb" />'
            f'<text x="{margin_left + scale_width + 16}" y="{y + 20}" '
            f'font-size="14" font-family="Arial">{value:.3f}</text>'
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            'viewBox="0 0 720 260" role="img" aria-label="Official NASA validation metrics">',
            '<rect width="720" height="260" fill="white" />',
            '<text x="20" y="24" font-size="16" font-weight="700" font-family="Arial">'
            "PulseNet official NASA FD001 metrics</text>",
            *bars,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def run_validation() -> dict[str, Any]:
    fd001 = load_official_fd001(
        "data/official", max_train_rows=None, max_test_rows=None
    )
    train_df = compute_rolling_features(fd001.train.copy())
    test_df = compute_rolling_features(fd001.test.copy())
    train_df, test_df, _ = normalize(train_df, test_df)
    feature_cols = get_feature_columns(train_df)
    x_test = test_df[feature_cols].to_numpy()
    y_test = create_labels(test_df, fd001.rul, failure_threshold=125)

    healthy_train = train_df[train_df["time_in_cycles"] <= 40][feature_cols].to_numpy()
    t0 = time.perf_counter()
    model = IsolationForestModel(n_estimators=100, contamination=0.12)
    model.train(healthy_train)
    train_seconds = time.perf_counter() - t0
    metrics = model.evaluate(x_test, y_test)

    results = {
        "dataset": {
            "name": "NASA C-MAPSS FD001",
            "source_url": fd001.source_url,
            "landing_page": fd001.landing_page,
            "archive_sha256": fd001.archive_sha256,
            "train_rows": int(len(fd001.train)),
            "test_rows": int(len(fd001.test)),
        },
        "labeling": {
            "failure_rul_threshold": 125,
            "positive_rate": round(float(y_test.mean()), 4),
        },
        "isolation_forest": {
            "f1": round(float(metrics["f1"]), 4),
            "precision": round(float(metrics["precision"]), 4),
            "recall": round(float(metrics["recall"]), 4),
            "roc_auc": round(float(metrics["roc_auc"]), 4),
            "train_time_s": round(float(train_seconds), 3),
            "features": int(len(feature_cols)),
        },
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/validation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    evidence_dir = Path("docs/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "validation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_metric_chart(
        {
            "F1": float(metrics["f1"]),
            "Precision": float(metrics["precision"]),
            "Recall": float(metrics["recall"]),
            "ROC AUC": float(metrics["roc_auc"]),
        },
        evidence_dir / "validation_metrics.svg",
    )
    return results


def main() -> None:
    results = run_validation()
    print(json.dumps(results, indent=2))
    print("Saved to results/validation_results.json")
    print("Saved to docs/evidence/validation_results.json")
    print("Saved to docs/evidence/validation_metrics.svg")


if __name__ == "__main__":
    main()
