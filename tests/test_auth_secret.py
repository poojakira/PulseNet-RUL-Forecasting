"""JWT secret-loading hardening tests."""

import importlib

import pytest

from pulsenet.api import auth


def test_get_jwt_secret_defers_missing_secret_until_called(monkeypatch):
    monkeypatch.delenv("PULSENET_JWT_SECRET", raising=False)
    monkeypatch.delenv("PULSENET_ENV", raising=False)
    auth.get_jwt_secret.cache_clear()

    with pytest.raises(RuntimeError, match="PULSENET_JWT_SECRET is not set"):
        auth.get_jwt_secret()


def test_get_jwt_secret_rejects_short_secret(monkeypatch):
    monkeypatch.setenv("PULSENET_JWT_SECRET", "short")
    auth.get_jwt_secret.cache_clear()

    with pytest.raises(ValueError, match=">=32 bytes"):
        auth.get_jwt_secret()


def test_auth_module_imports_without_secret(monkeypatch):
    monkeypatch.delenv("PULSENET_JWT_SECRET", raising=False)
    monkeypatch.delenv("PULSENET_ENV", raising=False)
    importlib.reload(auth)
    auth.get_jwt_secret.cache_clear()

    with pytest.raises(RuntimeError):
        auth.get_jwt_secret()