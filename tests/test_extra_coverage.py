"""Additional coverage tests: encryption, registry, audit, preprocessing, training."""

from __future__ import annotations

import pandas as pd
import pytest

from pulsenet.security.audit import AuditLogger
from pulsenet.security.encryption import EncryptionManager


class TestEncryptionManager:
    def test_generated_key_roundtrip(self, tmp_path):
        mgr = EncryptionManager(key_file=str(tmp_path / "k.key"))
        assert mgr._key_source == "generated"
        ct = mgr.encrypt("hello")
        assert mgr.decrypt(ct) == "hello"
        assert mgr.decrypt_bytes(mgr.encrypt_bytes(b"raw")) == b"raw"

    def test_key_loaded_from_file(self, tmp_path):
        kf = str(tmp_path / "k.key")
        EncryptionManager(key_file=kf)  # generates
        mgr2 = EncryptionManager(key_file=kf)
        assert mgr2._key_source == "file"

    def test_key_from_env(self, tmp_path, monkeypatch):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        monkeypatch.setenv("PULSENET_TEST_KEY", key)
        mgr = EncryptionManager(
            key_env_var="PULSENET_TEST_KEY", key_file=str(tmp_path / "k.key")
        )
        assert mgr._key_source == "environment"

    def test_dataframe_and_payload(self, tmp_path):
        mgr = EncryptionManager(key_file=str(tmp_path / "k.key"))
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        enc = mgr.encrypt_dataframe(df)
        dec = mgr.decrypt_dataframe(enc)
        assert dec.shape == df.shape

        ct = mgr.encrypt_payload({"x": 1, "y": "z"})
        assert mgr.decrypt_payload(ct) == {"x": 1, "y": "z"}

    def test_decrypt_cell(self, tmp_path):
        mgr = EncryptionManager(key_file=str(tmp_path / "k.key"))
        ct = mgr.encrypt("3.14")
        assert mgr.decrypt_cell(ct) == pytest.approx(3.14)
        assert mgr.decrypt_cell("not-a-cipher") == 0.0

    def test_rotate_key(self, tmp_path):
        mgr = EncryptionManager(key_file=str(tmp_path / "k.key"))
        old = mgr._key
        new = mgr.rotate_key()
        assert new != old
        assert mgr.decrypt(mgr.encrypt("after-rotation")) == "after-rotation"


class TestModelRegistryExtra:
    def test_lazy_load_all_model_types(self):
        from pulsenet.models.registry import ModelRegistry

        registry = ModelRegistry()
        assert registry.get_model("lstm").name == "lstm"
        assert registry.get_model("transformer").name == "transformer"
        assert registry.get_model("ensemble").name == "ensemble"
        assert set(["isolation_forest", "lstm", "transformer", "ensemble"]).issubset(
            set(registry.available_models)
        )

    def test_best_model(self, sample_X, sample_y):
        from pulsenet.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.get_model("isolation_forest").train(sample_X)
        best = registry.best_model(sample_X, sample_y, metric="f1")
        assert best == "isolation_forest"


class TestAuditLogger:
    def test_log_and_verify(self, tmp_path):
        logger = AuditLogger(log_file=str(tmp_path / "audit.jsonl"))
        h = logger.log_access("/predict", "POST", user="op", role="operator")
        assert len(h) == 64
        recent = logger.get_recent(n=10)
        assert len(recent) == 1
        valid, corrupt = logger.verify_integrity()
        assert valid is True
        assert corrupt == 0

    def test_tenant_isolation(self, tmp_path):
        logger = AuditLogger(log_file=str(tmp_path / "audit.jsonl"))
        logger.log_access("/x", "GET", tenant_id="acme")
        assert (tmp_path / "access_audit_acme.jsonl").exists()
        assert logger.get_recent(tenant_id="acme")

    def test_invalid_tenant_returns_failure(self, tmp_path):
        logger = AuditLogger(log_file=str(tmp_path / "audit.jsonl"))
        result = logger.log_access("/x", "GET", tenant_id="../escape")
        assert result == "ACCESS_LOG_FAILURE"

    def test_verify_missing_tenant(self, tmp_path):
        logger = AuditLogger(log_file=str(tmp_path / "audit.jsonl"))
        assert logger.verify_integrity(tenant_id="ghost") == (True, 0)
        assert logger.get_recent(tenant_id="ghost") == []


class TestPreprocessingExtra:
    def test_create_sequences_and_pipeline(self, official_fd001):
        from pulsenet.pipeline.preprocessing import (
            create_sequences,
            get_feature_columns,
            preprocess_pipeline,
        )

        train = official_fd001.train.copy()
        test = official_fd001.test.copy()
        tr, te, scaler = preprocess_pipeline(train, test, rolling_window=3)
        assert scaler is not None

        feat_cols = get_feature_columns(tr)
        seqs = create_sequences(tr, feat_cols, seq_len=5)
        assert seqs.ndim == 3

        empty = create_sequences(tr, feat_cols, seq_len=100000)
        assert empty.shape[0] == 0


class TestTrainingPipelineTune:
    def test_train_model_with_tuning(self, sample_X, sample_y, tmp_path):
        from pulsenet.models.training import TrainingPipeline

        pipeline = TrainingPipeline(model_dir=str(tmp_path / "models"))
        result = pipeline.train_model("isolation_forest", sample_X, sample_y, tune=True)
        assert result["model"] == "isolation_forest"
        model = pipeline.load_latest("isolation_forest")
        assert len(model.predict(sample_X)) == len(sample_X)
