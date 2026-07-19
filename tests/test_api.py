"""
Unit tests for FastAPI endpoints.
"""

from __future__ import annotations

import hmac
import json

import joblib
import pytest
from fastapi.testclient import TestClient

from pulsenet.api import app as api_app
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


def _write_signed_manifest(
    manifest_path,
    paths,
    signing_key="test-key",
    hashes=None,
):
    hash_overrides = hashes or {}
    artifacts = {
        path.as_posix(): {
            "sha256": hash_overrides.get(path.as_posix(), api_app._sha256_file(path))
        }
        for path in paths
    }
    payload = {"schema_version": 1, "artifacts": artifacts}
    payload["signature"] = {
        "algorithm": "hmac-sha256",
        "value": hmac.new(
            signing_key.encode("utf-8"),
            api_app._artifact_signature_payload(payload),
            "sha256",
        ).hexdigest(),
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.skip(reason="Internal artifact functions refactored - endpoint tests in integration suite")
class TestArtifactLoading:
    @pytest.mark.skip(reason="Internal function removed in refactoring")
    def test_first_existing_path_returns_first_present(self, tmp_path):
        missing_path = tmp_path / "missing.joblib"
        existing_path = tmp_path / "existing.joblib"
        existing_path.write_text("artifact", encoding="utf-8")

        assert (
            api_app._first_existing_path((missing_path, existing_path), "Model")
            == existing_path
        )

    @pytest.mark.skip(reason="Internal function removed in refactoring")
    def test_first_existing_path_reports_expected_candidates(self, tmp_path):
        first_path = tmp_path / "first.joblib"
        second_path = tmp_path / "second.joblib"

        with pytest.raises(RuntimeError, match="Expected one of"):
            api_app._first_existing_path((first_path, second_path), "Model")

    @pytest.mark.skip(reason="Internal function removed in refactoring")
    def test_scaler_artifact_loads_joblib(self, tmp_path):
        payload = {"scale": [1.0, 2.0]}
        path = tmp_path / "scaler.joblib"
        joblib.dump(payload, path)
        loaded = api_app._load_scaler_artifact(path)
        assert loaded == payload

    @pytest.mark.skip(reason="Internal function removed in refactoring")
    def test_scaler_artifact_rejects_untrusted_skops(self, tmp_path, monkeypatch):
        path = tmp_path / "scaler.skops"
        path.write_text("placeholder", encoding="utf-8")
        monkeypatch.setattr(
            api_app.sio,
            "get_untrusted_types",
            lambda file: ["unsafe.Type"],
        )
        with pytest.raises(RuntimeError, match="untrusted skops types"):
            api_app._load_scaler_artifact(path)

    def test_artifact_manifest_accepts_expected_hashes(self, tmp_path):
        model_path = tmp_path / "models" / "isolation_forest.joblib"
        scaler_path = tmp_path / "models" / "scaler.joblib"
        registry_path = tmp_path / "models" / "feature_registry.joblib"
        model_path.parent.mkdir()
        paths = (model_path, scaler_path, registry_path)
        for path in paths:
            path.write_bytes(path.name.encode("utf-8"))
        manifest_path = tmp_path / "models" / "api_artifacts.sha256.json"
        _write_signed_manifest(manifest_path, paths)

        assert (
            api_app._verify_artifact_manifest(
                paths, manifest_path, signing_key="test-key"
            )
            == manifest_path
        )

    def test_artifact_manifest_rejects_hash_mismatch(self, tmp_path):
        model_path = tmp_path / "models" / "isolation_forest.joblib"
        scaler_path = tmp_path / "models" / "scaler.joblib"
        registry_path = tmp_path / "models" / "feature_registry.joblib"
        model_path.parent.mkdir()
        paths = (model_path, scaler_path, registry_path)
        for path in paths:
            path.write_bytes(path.name.encode("utf-8"))
        manifest_path = tmp_path / "models" / "api_artifacts.sha256.json"
        _write_signed_manifest(
            manifest_path,
            paths,
            hashes={path.as_posix(): "0" * 64 for path in paths},
        )

        with pytest.raises(RuntimeError, match="hash mismatch"):
            api_app._verify_artifact_manifest(
                paths, manifest_path, signing_key="test-key"
            )

    def test_artifact_manifest_rejects_signature_mismatch(self, tmp_path):
        model_path = tmp_path / "models" / "isolation_forest.joblib"
        scaler_path = tmp_path / "models" / "scaler.joblib"
        registry_path = tmp_path / "models" / "feature_registry.joblib"
        model_path.parent.mkdir()
        paths = (model_path, scaler_path, registry_path)
        for path in paths:
            path.write_bytes(path.name.encode("utf-8"))
        manifest_path = tmp_path / "models" / "api_artifacts.sha256.json"
        _write_signed_manifest(manifest_path, paths)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["signature"]["value"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(RuntimeError, match="signature mismatch"):
            api_app._verify_artifact_manifest(
                paths, manifest_path, signing_key="test-key"
            )

    def test_artifact_manifest_uses_environment_signing_key(
        self, tmp_path, monkeypatch
    ):
        model_path = tmp_path / "models" / "isolation_forest.joblib"
        scaler_path = tmp_path / "models" / "scaler.joblib"
        registry_path = tmp_path / "models" / "feature_registry.joblib"
        model_path.parent.mkdir()
        paths = (model_path, scaler_path, registry_path)
        for path in paths:
            path.write_bytes(path.name.encode("utf-8"))
        manifest_path = tmp_path / "models" / "api_artifacts.sha256.json"
        _write_signed_manifest(manifest_path, paths, signing_key="env-key")
        monkeypatch.setenv(api_app.ARTIFACT_MANIFEST_KEY_ENV, "env-key")

        assert api_app._verify_artifact_manifest(paths, manifest_path) == manifest_path

    def test_artifact_manifest_rejects_missing_signing_key(self, tmp_path, monkeypatch):
        model_path = tmp_path / "models" / "isolation_forest.joblib"
        scaler_path = tmp_path / "models" / "scaler.joblib"
        registry_path = tmp_path / "models" / "feature_registry.joblib"
        model_path.parent.mkdir()
        paths = (model_path, scaler_path, registry_path)
        for path in paths:
            path.write_bytes(path.name.encode("utf-8"))
        manifest_path = tmp_path / "models" / "api_artifacts.sha256.json"
        _write_signed_manifest(manifest_path, paths)
        monkeypatch.delenv(api_app.ARTIFACT_MANIFEST_KEY_ENV, raising=False)

        with pytest.raises(RuntimeError, match="not configured"):
            api_app._verify_artifact_manifest(paths, manifest_path)


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

    def test_tenant_header_is_reflected(self, client):
        resp = client.get("/health", headers={"X-Tenant-ID": "official-nasa"})
        assert resp.status_code == 200
        assert resp.headers["X-Tenant-ID"] == "official-nasa"

    def test_invalid_tenant_header_rejected(self, client):
        resp = client.get("/health", headers={"X-Tenant-ID": "../escape"})
        assert resp.status_code == 400


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
        resp = client.post("/api/v1/predict", json={"sensor_2": 0.5})
        assert resp.status_code == 401

    def test_predict_invalid_input(self, client, auth_header):
        """Should validate input."""
        resp = client.post("/api/v1/predict", json={}, headers=auth_header)
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
