from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from pulsenet.security.encryption import EncryptionManager


def test_invalid_configured_key_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PULSENET_ENCRYPTION_KEY", "not-a-valid-fernet-key")

    with pytest.raises(RuntimeError, match="not a valid Fernet key"):
        EncryptionManager(key_file=tmp_path / "unused.key")


def test_tampered_ciphertext_never_becomes_sensor_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PULSENET_ENCRYPTION_KEY", Fernet.generate_key().decode())
    manager = EncryptionManager(key_file=tmp_path / "unused.key")
    ciphertext = manager.encrypt("42.5")
    tampered = ciphertext[:-1] + ("A" if ciphertext[-1] != "A" else "B")

    with pytest.raises(ValueError, match="authentication"):
        manager.decrypt_cell(tampered)


def test_non_numeric_plaintext_fails_instead_of_defaulting_to_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PULSENET_ENCRYPTION_KEY", Fernet.generate_key().decode())
    manager = EncryptionManager(key_file=tmp_path / "unused.key")

    with pytest.raises(ValueError, match="not a valid numeric value"):
        manager.decrypt_cell(manager.encrypt("sensor-error"))
