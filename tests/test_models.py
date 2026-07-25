"""
Unit tests for ML models — Isolation Forest, registry, training pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.models.registry import ModelRegistry
from pulsenet.models.training import TrainingPipeline


class TestIsolationForest:
    """Tests for Isolation Forest model."""

    def test_train_predict(self, sample_x, sample_y):
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.train(sample_x)
        preds = model.predict(sample_x)
        assert len(preds) == len(sample_x)
        assert set(preds).issubset({0, 1})

    def test_score(self, sample_x):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_x)
        scores = model.score(sample_x)
        assert len(scores) == len(sample_x)
        assert scores.dtype == np.float64

    def test_health_index(self, sample_x):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_x)
        health = model.health_index(sample_x)
        assert all(0 <= h <= 100 for h in health)

    def test_save_load(self, sample_x, temp_dir):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_x)
        preds_before = model.predict(sample_x)

        path = temp_dir / "test_model.joblib"
        model.save(path)

        loaded = IsolationForestModel()
        loaded.load(path)
        preds_after = loaded.predict(sample_x)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_evaluate(self, sample_x, sample_y):
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.train(sample_x)
        metrics = model.evaluate(sample_x, sample_y)
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_tune(self, sample_x, sample_y):
        model = IsolationForestModel()
        result = model.tune(
            sample_x,
            sample_y,
            n_estimators_list=[50, 100],
            contamination_list=[0.05, 0.1],
            max_samples_list=[0.8],
        )
        assert "best_f1" in result
        assert "best_params" in result
        assert result["best_f1"] >= 0

    def test_optimize_threshold(self, sample_x, sample_y):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_x)
        threshold = model.optimize_threshold(sample_x, sample_y)
        assert isinstance(threshold, float)
        assert model.threshold == threshold


class TestModelRegistry:
    """Tests for model registry."""

    def test_register_and_get(self):
        registry = ModelRegistry()
        assert "isolation_forest" in registry.available_models

    def test_get_unknown(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get_model("nonexistent")

    def test_compare_all(self, sample_x, sample_y):
        registry = ModelRegistry()
        model = registry.get_model("isolation_forest")
        model.train(sample_x)
        results = registry.compare_all(sample_x, sample_y)
        assert "isolation_forest" in results


class TestTrainingPipeline:
    """Tests for training pipeline."""

    def test_train_model(self, sample_x, sample_y, temp_dir):
        pipeline = TrainingPipeline(model_dir=str(temp_dir))
        result = pipeline.train_model("isolation_forest", sample_x, sample_y)
        assert result["model"] == "isolation_forest"
        assert "version" in result
        assert result["train_time_sec"] >= 0

    def test_load_latest(self, sample_x, temp_dir):
        pipeline = TrainingPipeline(model_dir=str(temp_dir))
        pipeline.train_model("isolation_forest", sample_x)
        model = pipeline.load_latest("isolation_forest")
        preds = model.predict(sample_x)
        assert len(preds) == len(sample_x)
