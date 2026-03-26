"""
Unit tests for FastAPI endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pulsenet.api.app import create_app
from pulsenet.api.auth import create_token


@pytest.fixture
def client():
    """Test client with fresh app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def admin_token():
    token, _ = create_token("admin", "admin")
    return token


@pytest.fixture
def operator_token():
    token, _ = create_token("operator", "operator")
    return token


@pytest.fixture
def auth_header(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_no_auth(self, client):
        """Health endpoint should work without auth."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "2.1.0"

    def test_health_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "model_loaded" in data
        assert "blockchain_blocks" in data
        assert "uptime_seconds" in data
        assert "gpu_devices" in data
        assert "system_resources" in data


class TestAuthEndpoint:
    """Test /token endpoint."""

    def test_valid_login(self, client):
        # Admin default password is now admin123 (hashed in auth.py)
        resp = client.post("/token", json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["role"] == "admin"

    def test_invalid_login(self, client):
        resp = client.post("/token", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_operator_login(self, client):
        resp = client.post(
            "/token", json={"username": "operator", "password": "ops123"}
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "operator"


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_no_auth(self, client):
        """Should require authentication."""
        resp = client.post("/predict", json={"sensor_2": 0.5})
        assert resp.status_code == 401

    def test_predict_invalid_input(self, client, auth_header):
        """Should validate input."""
        resp = client.post("/predict", json={}, headers=auth_header)
        assert resp.status_code == 422  # Validation error


class TestAuditEndpoint:
    """Test /audit endpoint."""

    def test_audit_requires_auth(self, client):
        resp = client.get("/audit")
        assert resp.status_code == 401

    def test_audit_with_auth(self, client, auth_header):
        resp = client.get("/audit", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "chain_length" in data
        assert "is_valid" in data

    def test_audit_rbac(self, client, operator_token):
        """Operators should not access audit endpoint."""
        headers = {"Authorization": f"Bearer {operator_token}"}
        resp = client.get("/audit", headers=headers)
        assert resp.status_code == 403


class TestVerifyChainEndpoint:
    """Test /verify-chain endpoint."""

    def test_verify_requires_auth(self, client):
        resp = client.get("/verify-chain")
        assert resp.status_code == 401

    def test_verify_with_auth(self, client, auth_header):
        resp = client.get("/verify-chain", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "valid" in data
        assert "message" in data
