"""
Unit tests for security module — encryption, blockchain, audit logging.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from pulsenet.security.encryption import EncryptionManager
from pulsenet.security.blockchain import BlackBoxLedger, Block
from pulsenet.security.audit import AuditLogger


class TestEncryptionManager:
    """Tests for AES-256 encryption."""

    @pytest.fixture
    def enc(self, temp_dir):
        key_file = temp_dir / "test.key"
        return EncryptionManager(key_file=str(key_file))

    def test_encrypt_decrypt_roundtrip(self, enc):
        plaintext = "sensor_value_123.456"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = enc.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_encrypt_decrypt_bytes(self, enc):
        data = b"binary_sensor_data"
        encrypted = enc.encrypt_bytes(data)
        assert encrypted != data
        decrypted = enc.decrypt_bytes(encrypted)
        assert decrypted == data

    def test_encrypt_dataframe(self, enc, sample_sensor_data):
        df = sample_sensor_data.head(5)
        encrypted = enc.encrypt_dataframe(df)
        assert encrypted.shape == df.shape
        # Values should all be different (encrypted)
        assert not (encrypted.iloc[0, 0] == df.iloc[0, 0])

    def test_key_rotation(self, enc, temp_dir):
        key_file = temp_dir / "test.key"
        old_key = key_file.read_bytes()
        enc.rotate_key()
        new_key = key_file.read_bytes()
        assert old_key != new_key
        # Backup should exist
        assert (temp_dir / "test.key.bak").exists()

    def test_encrypt_payload(self, enc):
        payload = {"sensor_2": 0.5, "status": "healthy"}
        encrypted = enc.encrypt_payload(payload)
        decrypted = enc.decrypt_payload(encrypted)
        assert decrypted == payload

    def test_decrypt_cell(self, enc):
        val = enc.encrypt("42.5")
        result = enc.decrypt_cell(val)
        assert result == 42.5


class TestBlockchain:
    """Tests for blockchain ledger."""

    @pytest.fixture
    def ledger(self, temp_dir):
        return BlackBoxLedger(chain_file=str(temp_dir / "test_chain.json"))

    def test_genesis_block(self, ledger):
        assert len(ledger.chain) == 1
        assert ledger.chain[0].index == 0
        assert ledger.chain[0].data == "GENESIS_BLOCK_ENGINE_START"

    def test_add_entry(self, ledger):
        hash_val = ledger.add_entry(unit_id=1, cycles=100, health_score=85.5, status="OPTIMAL")
        assert len(hash_val) == 64  # SHA-256 hex
        assert len(ledger.chain) == 2

    def test_validate_integrity(self, ledger):
        ledger.add_entry(1, 100, 85.5, "OPTIMAL")
        ledger.add_entry(1, 101, 82.3, "WARNING")
        is_valid, msg = ledger.validate_integrity()
        assert is_valid
        assert "verified" in msg.lower()

    def test_tamper_detection(self, ledger):
        ledger.add_entry(1, 100, 85.5, "OPTIMAL")
        # Tamper with data
        ledger.chain[1].data = {"unit_id": 1, "cycles": 999, "health_score": 0, "status": "FAKE"}
        is_valid, msg = ledger.validate_integrity()
        assert not is_valid
        tampered = ledger.detect_tampering()
        assert 1 in tampered

    def test_merkle_root(self, ledger):
        ledger.add_entry(1, 100, 85.5, "OPTIMAL")
        root = ledger.compute_merkle_root()
        assert len(root) == 64

    def test_chain_persistence(self, temp_dir):
        file = str(temp_dir / "persist_chain.json")
        l1 = BlackBoxLedger(file)
        l1.add_entry(1, 100, 85.5, "OPTIMAL")
        hash1 = l1.chain[1].hash

        l2 = BlackBoxLedger(file)
        assert len(l2.chain) == 2
        assert l2.chain[1].hash == hash1

    def test_get_metrics(self, ledger):
        ledger.add_entry(1, 100, 85.5, "OPTIMAL")
        metrics = ledger.get_metrics()
        assert "total_blocks" in metrics
        assert "chain_valid" in metrics
        assert metrics["chain_valid"] is True

    def test_get_recent_blocks(self, ledger):
        for i in range(5):
            ledger.add_entry(1, i, 80.0, "OPTIMAL")
        blocks = ledger.get_recent_blocks(3)
        assert len(blocks) == 3


class TestAuditLogger:
    """Tests for access audit logging."""

    @pytest.fixture
    def audit(self, temp_dir):
        return AuditLogger(log_file=str(temp_dir / "test_audit.jsonl"))

    def test_log_access(self, audit):
        hash_val = audit.log_access(
            endpoint="/predict", method="POST",
            user="admin", role="admin",
        )
        assert len(hash_val) == 64

    def test_get_recent(self, audit):
        audit.log_access("/health", "GET", "user1", "operator")
        audit.log_access("/predict", "POST", "user2", "engineer")
        entries = audit.get_recent()
        assert len(entries) == 2
        assert entries[0]["endpoint"] == "/health"

    def test_verify_integrity(self, audit):
        audit.log_access("/health", "GET", "admin", "admin")
        audit.log_access("/predict", "POST", "user1", "engineer")
        is_valid, corrupt = audit.verify_integrity()
        assert is_valid
        assert corrupt == 0
