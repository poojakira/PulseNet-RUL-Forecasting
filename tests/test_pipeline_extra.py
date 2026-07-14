"""Tests for adversarial telemetry guard, pipeline components, and misc modules."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from pulsenet.core.exceptions import (
    ConfigurationError,
    DataError,
    ModelError,
    PulseNetError,
    SecurityError,
)
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.security.adversarial_telemetry import AdversarialTelemetryGuard


class TestAdversarialTelemetryGuard:
    def test_from_scaler_variants(self):
        sc = MinMaxScaler().fit(np.random.default_rng(0).standard_normal((50, 4)))
        guard = AdversarialTelemetryGuard.from_scaler(sc)
        assert guard.reference_mean is not None

        assert AdversarialTelemetryGuard.from_scaler(None).reference_mean is None

        class _NoBounds:
            pass

        assert AdversarialTelemetryGuard.from_scaler(_NoBounds()).reference_mean is None

    def test_fit_and_assess(self):
        rng = np.random.default_rng(1)
        x_ref = rng.standard_normal((100, 4))
        guard = AdversarialTelemetryGuard(
            ood_z_threshold=4.0, max_rows_for_perturbation=3
        )
        guard.fit_reference(x_ref)

        model = IsolationForestModel(n_estimators=20)
        model.train(x_ref)

        res = guard.assess(model, rng.standard_normal((6, 4)))
        assert res.sampled_rows == 3
        assert res.max_perturbation_delta >= 0.0

        ood = guard.assess(model, rng.standard_normal((6, 4)) * 30 + 100)
        assert ood.ood_detected is True

    def test_assess_edge_cases(self):
        guard = AdversarialTelemetryGuard()
        model = IsolationForestModel(n_estimators=10)
        model.train(np.random.default_rng(2).standard_normal((20, 4)))

        empty = guard.assess(model, np.zeros((0, 4)))
        assert empty.sampled_rows == 0

        one_d = guard.assess(model, np.zeros(4))
        assert one_d.sampled_rows == 0


class TestExceptions:
    def test_all_exceptions(self):
        for exc_cls in (DataError, ModelError, SecurityError, ConfigurationError):
            err = exc_cls("boom")
            assert isinstance(err, PulseNetError)
            assert err.message == "boom"
            assert err.error_code

        base = PulseNetError("x", "CODE")
        assert base.error_code == "CODE"


class TestFeatureRegistry:
    def test_offline_online_roundtrip(self, sample_sensor_data):
        from pulsenet.pipeline.feature_registry import FeatureRegistry

        fr = FeatureRegistry(rolling_window=3)
        names = fr.get_feature_names([str(c) for c in sample_sensor_data.columns])
        assert any(n.endswith("_rolling_mean") for n in names)

        processed = fr.process_offline(sample_sensor_data.copy())
        assert fr.feature_cols
        fr.fit_scaler(processed)
        assert fr.is_fitted

        config = fr.save_config()
        restored = FeatureRegistry()
        restored.load_config(config)
        assert restored.rolling_window == 3

        row = {c: 0.5 for c in fr.feature_cols}
        with_hist = fr.process_online(dict(row), history=processed.head(5))
        assert with_hist.shape[0] == 1
        no_hist = fr.process_online(dict(row), history=None)
        assert no_hist.shape[0] == 1

    def test_process_online_unfitted_raises(self):
        from pulsenet.pipeline.feature_registry import FeatureRegistry

        fr = FeatureRegistry()
        with pytest.raises(DataError):
            fr.process_online({"sensor_2": 1.0})


class TestIngestion:
    def test_load_and_ingest(self, official_fd001):
        from pulsenet.pipeline import ingestion
        from pulsenet.pipeline.official_cmapss import _find_file

        root = Path(__file__).resolve().parents[1] / "data" / "official" / "CMAPSSData"
        train_p = _find_file(root, "train_FD001.txt")
        test_p = _find_file(root, "test_FD001.txt")
        rul_p = _find_file(root, "RUL_FD001.txt")

        df = ingestion.load_raw(train_p)
        assert len(df.columns) == 26
        rul = ingestion.load_rul(rul_p)
        assert len(rul) > 0

        dropped = ingestion.drop_noisy_columns(df, ["sensor_1", "op_setting_1"])
        assert "sensor_1" not in dropped.columns

        train_df, test_df = ingestion.ingest(train_p, test_p, drop_cols=["sensor_1"])
        assert "sensor_1" not in train_df.columns
        assert len(test_df) > 0

    def test_missing_files_raise(self):
        from pulsenet.pipeline import ingestion

        with pytest.raises(DataError):
            ingestion.load_raw("does_not_exist.txt")
        with pytest.raises(DataError):
            ingestion.load_rul("does_not_exist_rul.txt")


class TestOrchestrator:
    def test_full_pipeline_isolation_forest(self, tmp_path, cmapss_zip):
        from pulsenet.pipeline.orchestrator import PipelineOrchestrator

        official = tmp_path / "official"
        official.mkdir(parents=True)
        shutil.copy(cmapss_zip, official / "CMAPSSData.zip")

        orch = PipelineOrchestrator(data_dir=str(tmp_path))
        orch.run_ingestion()
        assert orch.train_df is not None and orch.test_df is not None
        orch.run_preprocessing()
        assert orch.scaler is not None
        orch.run_training("isolation_forest")
        results = orch.run_evaluation()
        assert "isolation_forest" in results
        infer_df = orch.run_inference("isolation_forest")
        assert "prediction" in infer_df.columns


class TestTrainingPipelineExtra:
    def test_train_all(self, sample_X, tmp_path):
        from pulsenet.models.training import TrainingPipeline

        pipeline = TrainingPipeline(model_dir=str(tmp_path / "models"))
        results = pipeline.train_all(sample_X)
        assert any(r.get("model") == "isolation_forest" for r in results)
