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

    def test_train_predict(self, sample_X, sample_y):
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.train(sample_X)
        preds = model.predict(sample_X)
        assert len(preds) == len(sample_X)
        assert set(preds).issubset({0, 1})

    def test_score(self, sample_X):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_X)
        scores = model.score(sample_X)
        assert len(scores) == len(sample_X)
        assert scores.dtype == np.float64

    def test_health_index(self, sample_X):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_X)
        health = model.health_index(sample_X)
        assert all(0 <= h <= 100 for h in health)

    def test_save_load(self, sample_X, temp_dir):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_X)
        preds_before = model.predict(sample_X)

        path = temp_dir / "test_model.joblib"
        model.save(path)

        loaded = IsolationForestModel()
        loaded.load(path)
        preds_after = loaded.predict(sample_X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_evaluate(self, sample_X, sample_y):
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.train(sample_X)
        metrics = model.evaluate(sample_X, sample_y)
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_tune(self, sample_X, sample_y):
        model = IsolationForestModel()
        result = model.tune(
            sample_X,
            sample_y,
            n_estimators_list=[50, 100],
            contamination_list=[0.05, 0.1],
            max_samples_list=[0.8],
        )
        assert "best_f1" in result
        assert "best_params" in result
        assert result["best_f1"] >= 0

    def test_optimize_threshold(self, sample_X, sample_y):
        model = IsolationForestModel(n_estimators=50)
        model.train(sample_X)
        threshold = model.optimize_threshold(sample_X, sample_y)
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

    def test_compare_all(self, sample_X, sample_y):
        registry = ModelRegistry()
        model = registry.get_model("isolation_forest")
        model.train(sample_X)
        results = registry.compare_all(sample_X, sample_y)
        assert "isolation_forest" in results


class TestTrainingPipeline:
    """Tests for training pipeline."""

    def test_train_model(self, sample_X, sample_y, temp_dir):
        pipeline = TrainingPipeline(model_dir=str(temp_dir))
        result = pipeline.train_model("isolation_forest", sample_X, sample_y)
        assert result["model"] == "isolation_forest"
        assert "version" in result
        assert result["train_time_sec"] >= 0

    def test_load_latest(self, sample_X, temp_dir):
        pipeline = TrainingPipeline(model_dir=str(temp_dir))
        pipeline.train_model("isolation_forest", sample_X)
        model = pipeline.load_latest("isolation_forest")
        preds = model.predict(sample_X)
        assert len(preds) == len(sample_X)
