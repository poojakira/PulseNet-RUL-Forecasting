"""Targeted tests to cover remaining branches (ingestion, registry, mlflow backend)."""

from __future__ import annotations

import numpy as np

from pulsenet.mlops.tracker import MLOpsTracker
from pulsenet.models.registry import ModelRegistry


class TestIngestionValidation:
    def test_nan_and_inf_are_cleaned(self, tmp_path):
        from pulsenet.pipeline import ingestion

        # 26 whitespace-separated columns per row; inject NaN and Inf.
        def _row(unit, cycle, bad_idx=None, bad="nan"):
            vals = [str(unit), str(cycle)] + [f"{v:.3f}" for v in range(24)]
            if bad_idx is not None:
                vals[bad_idx] = bad
            return " ".join(vals)

        content = "\n".join(
            [
                _row(1, 1, bad_idx=5, bad="nan"),
                _row(1, 2, bad_idx=6, bad="inf"),
                _row(1, 3),
            ]
        )
        path = tmp_path / "train_FD001.txt"
        path.write_text(content + "\n")

        df = ingestion.load_raw(path)
        assert len(df) == 3
        numeric = df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy()).any()
        assert int(df.isna().sum().sum()) == 0

    def test_ingest_uses_config_defaults(self, official_fd001):
        from pathlib import Path

        from pulsenet.pipeline import ingestion
        from pulsenet.pipeline.official_cmapss import _find_file

        root = Path(__file__).resolve().parents[1] / "data" / "official" / "CMAPSSData"
        train_p = _find_file(root, "train_FD001.txt")
        test_p = _find_file(root, "test_FD001.txt")
        # No drop_cols -> exercises the cfg-based default branch
        train_df, test_df = ingestion.ingest(train_p, test_p)
        assert len(train_df) > 0 and len(test_df) > 0


class TestRegistryErrorBranch:
    def test_compare_all_handles_untrained(self, sample_X, sample_y):
        registry = ModelRegistry()
        # isolation_forest is registered but NOT trained -> evaluate raises,
        # exercising the error-capture branch in compare_all.
        results = registry.compare_all(sample_X, sample_y)
        assert "error" in results["isolation_forest"]


class TestMlflowBackend:
    def test_mlflow_logging_paths(self, tmp_path):
        uri = (tmp_path / "mlruns").as_uri()
        tracker = MLOpsTracker(tracking_uri=uri, experiment_name="pytest-exp")
        if not tracker._mlflow_available:
            # Environment can't init MLflow file store; local path already covered.
            return
        run_id = tracker.log_training_run({"n_estimators": 50}, {"f1": 0.8})
        assert isinstance(run_id, str) and run_id
        tracker.log_inference_metrics(np.array([0, 1, 1]), latency_ms=3.0, batch_size=3)
