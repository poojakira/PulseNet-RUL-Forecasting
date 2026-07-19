"""Tests for the PyTorch-based models (LSTM, Transformer) and the Ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pulsenet.models.base import BaseAnomalyModel
from pulsenet.models.ensemble import EnsembleModel
from pulsenet.models.lstm_model import LSTMModel
from pulsenet.models.transformer_model import TransformerModel


def _seq_data(n: int = 24, seq: int = 5, feat: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, seq, feat)).astype("float32")


class TestLSTMModel:
    def test_train_predict_score_health(self):
        x = _seq_data()
        m = LSTMModel(
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            sequence_length=5,
            epochs=1,
            batch_size=8,
        )
        m.train(x)
        assert m.threshold is not None
        preds = m.predict(x)
        assert len(preds) == len(x)
        assert set(np.unique(preds)).issubset({0, 1})
        assert len(m.score(x)) == len(x)
        assert len(m.decision_function(x)) == len(x)
        health = m.health_index(x)
        assert np.all((health >= 0) & (health <= 100))

    def test_save_load_roundtrip(self, temp_dir):
        x = _seq_data()
        m = LSTMModel(
            hidden_size=8, num_layers=1, dropout=0.0, sequence_length=5, epochs=1
        )
        m.train(x)
        before = m.score(x)
        path = temp_dir / "lstm.pt"
        m.save(path)

        m2 = LSTMModel(
            hidden_size=8, num_layers=1, dropout=0.0, sequence_length=5, epochs=1
        )
        m2.load(path)
        np.testing.assert_allclose(before, m2.score(x), rtol=1e-3, atol=1e-3)

    def test_untrained_score_raises(self):
        with pytest.raises(RuntimeError):
            LSTMModel(epochs=1).score(_seq_data())

    def test_predict_without_threshold_raises(self):
        with pytest.raises(ValueError):
            LSTMModel(epochs=1).predict(_seq_data())

    def test_train_requires_3d(self):
        with pytest.raises(ValueError):
            LSTMModel(epochs=1).train(np.zeros((10, 4), dtype="float32"))

    def test_window_flat(self):
        flat = np.arange(20).reshape(10, 2).astype("float32")
        seqs = LSTMModel._window_flat(flat, 3)
        assert seqs.shape == (8, 3, 2)

    def test_health_index_zero_threshold(self):
        x = _seq_data()
        m = LSTMModel(hidden_size=8, num_layers=1, dropout=0.0, epochs=1)
        m.train(x)
        m.threshold = 0.0
        assert np.allclose(m.health_index(x), 100.0)


class TestTransformerModel:
    def test_train_predict_score_health(self):
        x = _seq_data()
        m = TransformerModel(
            d_model=8, nhead=2, num_layers=1, dropout=0.1, epochs=1, batch_size=8
        )
        m.train(x)
        assert m.threshold is not None
        preds = m.predict(x)
        assert len(preds) == len(x)
        assert len(m.score(x)) == len(x)
        assert len(m.decision_function(x)) == len(x)
        health = m.health_index(x)
        assert np.all((health >= 0) & (health <= 100))

    def test_save_load_roundtrip(self, temp_dir):
        x = _seq_data()
        m = TransformerModel(d_model=8, nhead=2, num_layers=1, epochs=1)
        m.train(x)
        before = m.score(x)
        path = temp_dir / "tf.pt"
        m.save(path)

        m2 = TransformerModel(d_model=8, nhead=2, num_layers=1, epochs=1)
        m2.load(path)
        np.testing.assert_allclose(before, m2.score(x), rtol=1e-3, atol=1e-3)

    def test_compute_errors_requires_3d(self):
        m = TransformerModel(d_model=8, nhead=2, num_layers=1, epochs=1)
        m.train(_seq_data())
        with pytest.raises(ValueError):
            m.score(np.zeros((10, 4), dtype="float32"))

    def test_window_flat(self):
        flat = np.arange(12).reshape(6, 2).astype("float32")
        assert TransformerModel._window_flat(flat, 2).shape == (5, 2, 2)


class _FakeSubModel(BaseAnomalyModel):
    """Deterministic lightweight sub-model to exercise ensemble logic."""

    def __init__(self, name: str, score_val: float = 0.5):
        self.name = name
        self._sv = score_val
        self.trained = False
        self.loaded = False

    def train(self, X, **kwargs):  # noqa: N803
        self.trained = True

    def predict(self, X):  # noqa: N803
        return (np.arange(len(X)) % 2).astype(int)

    def score(self, X):  # noqa: N803
        return np.full(len(X), self._sv)

    def decision_function(self, X):  # noqa: N803
        return self.score(X)

    def health_index(self, X):  # noqa: N803
        return np.full(len(X), 50.0)

    def save(self, path):
        Path(path).write_text("fake")

    def load(self, path):
        self.loaded = True


def _ensemble_with_fakes(strategy: str = "majority_vote") -> EnsembleModel:
    ens = EnsembleModel(strategy=strategy, threshold=0.5)
    ens._sub_models = [
        _FakeSubModel("a", 0.2),
        _FakeSubModel("b", 0.5),
        _FakeSubModel("c", 0.8),
    ]
    ens._model_names = [m.name for m in ens._sub_models]
    return ens


class TestEnsembleModel:
    def test_majority_vote_predict(self):
        ens = _ensemble_with_fakes("majority_vote")
        ens.train(np.zeros((6, 3)))
        assert all(m.trained for m in ens._sub_models)
        preds = ens.predict(np.zeros((6, 3)))
        assert len(preds) == 6
        assert set(np.unique(preds)).issubset({0, 1})

    def test_weighted_score_predict_and_score(self):
        ens = _ensemble_with_fakes("weighted_score")
        x = np.zeros((5, 3))
        scores = ens.score(x)
        assert len(scores) == 5
        preds = ens.predict(x)
        assert len(preds) == 5
        assert len(ens.decision_function(x)) == 5
        health = ens.health_index(x)
        assert np.all((health >= 0) & (health <= 100))

    def test_save_and_load(self, temp_dir):
        ens = _ensemble_with_fakes()
        path = temp_dir / "models" / "ensemble.joblib"
        ens.save(path)
        cfg_path = temp_dir / "models" / "ensemble" / "ensemble_config.json"
        assert cfg_path.exists()
        assert json.loads(cfg_path.read_text())["strategy"] == "majority_vote"

        ens2 = EnsembleModel()
        ens2.load(path)  # loads config + resets to real sub-models
        assert ens2.strategy == "majority_vote"

    def test_load_sub_models_lazy(self):
        ens = EnsembleModel()
        ens._load_sub_models()
        assert ens._model_names == ["isolation_forest", "lstm", "transformer"]
        assert ens.weights is not None and len(ens.weights) == 3
