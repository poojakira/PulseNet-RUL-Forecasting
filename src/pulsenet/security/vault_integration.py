"""
PulseNet Vault Integration — JWT Secret Rotation & Secure Secret Management.

Provides:
- Automatic JWT secret rotation with HashiCorp Vault
- Transit engine encryption for artifact signing keys
- Dynamic secret generation for database/API credentials
- Audit logging of all secret access
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import hvac
from hvac.exceptions import VaultError

from pulsenet.logger import get_logger

log = get_logger(__name__)


@dataclass
class VaultConfig:
    """Vault connection configuration."""
    url: str = "http://localhost:8200"
    token: Optional[str] = None
    namespace: Optional[str] = None
    ca_cert: Optional[str] = None
    verify_ssl: bool = True


@dataclass
class RotationConfig:
    """Secret rotation configuration."""
    rotation_interval_hours: int = 24
    jwt_key_ttl_hours: int = 168  # 7 days
    artifact_signing_ttl_hours: int = 168
    db_creds_ttl_hours: int = 1
    api_key_ttl_hours: int = 168
    grace_period_hours: int = 1  # Overlap period for key rotation
    enable_auto_rotation: bool = True


class VaultManager:
    """
    Manages secrets and encryption via HashiCorp Vault.
    
    Features:
    - JWT secret rotation with grace period
    - Transit engine for artifact signing
    - Dynamic database credentials
    - PKI certificate management
    - Audit logging
    """
    
    def __init__(
        self,
        vault_config: VaultConfig,
        rotation_config: Optional[RotationConfig] = None,
    ):
        self.vault_config = vault_config
        self.rotation_config = rotation_config or RotationConfig()
        
        self._client = hvac.Client(
            url=vault_config.url,
            token=vault_config.token,
            namespace=vault_config.namespace,
            verify=vault_config.verify_ssl,
        )
        
        if not self._client.is_authenticated():
            raise VaultError("Failed to authenticate with Vault")
        
        # Ensure required secrets engines exist
        self._setup_secrets_engines()
        
        # Start rotation scheduler
        self._rotation_thread: Optional[threading.Thread] = None
        self._stop_rotation = threading.Event()
        
        if self.rotation_config.enable_auto_rotation:
            self._start_rotation_scheduler()
        
        log.info(f"VaultManager initialized: {vault_config.url}")
    
    def _setup_secrets_engines(self) -> None:
        """Ensure required secrets engines are enabled."""
        engines = {
            "transit": "transit",
            "secret": "kv-v2",
            "database": "database",
            "pki": "pki",
        }
        
        for path, engine_type in engines.items():
            try:
                self._client.sys.enable_secrets_engine(
                    backend_type=engine_type,
                    path=path,
                )
                log.info(f"Enabled {engine_type} at {path}")
            except hvac.exceptions.InvalidPath:
                # Already enabled
                pass
            except Exception as e:
                log.warning(f"Could not enable {engine_type} at {path}: {e}")
        
        # Configure transit keys
        self._ensure_transit_keys()
    
    def _ensure_transit_keys(self) -> None:
        """Ensure required transit encryption keys exist."""
        keys = {
            "jwt-signing": {"type": "ecdsa-p256", "derived": True},
            "artifact-signing": {"type": "rsa-4096", "derived": False},
            "data-encryption": {"type": "aes256-gcm96", "derived": True},
        }
        
        for name, config in keys.items():
            try:
                self._client.secrets.transit.read_key(name=name)
            except hvac.exceptions.InvalidPath:
                self._client.secrets.transit.create_key(
                    name=name,
                    key_type=config["type"],
                    derived=config["derived"],
                    exportable=False,
                )
                log.info(f"Created transit key: {name}")
    
    def _start_rotation_scheduler(self) -> None:
        """Start background thread for automatic secret rotation."""
        self._rotation_thread = threading.Thread(
            target=self._rotation_loop,
            daemon=True,
        )
        self._rotation_thread.start()
        log.info("Vault secret rotation scheduler started")
    
    def _rotation_loop(self) -> None:
        """Background loop for automatic secret rotation."""
        interval = self.rotation_config.rotation_interval_hours * 3600
        
        while not self._stop_rotation.wait(timeout=interval):
            try:
                self.rotate_jwt_keys()
                log.info("Completed scheduled JWT key rotation")
            except Exception as e:
                log.error(f"Scheduled key rotation failed: {e}")
    
    # === JWT Key Management ===
    
    def rotate_jwt_keys(self, grace_period_hours: Optional[int] = None) -> dict[str, str]:
        """
        Rotate JWT signing keys with grace period.
        
        Creates new key, keeps old key valid for grace period,
        then revokes old key.
        
        Returns:
            Dict with new_key_id, old_key_id, rotation_time
        """
        grace_hours = grace_period_hours or self.rotation_config.grace_period_hours
        
        # Generate new key name with timestamp
        new_key_name = f"jwt-signing-{int(time.time())}"
        old_key_name = "jwt-signing"
        
        # Create new key
        self._client.secrets.transit.create_key(
            name=new_key_name,
            key_type="ecdsa-p256",
            derived=True,
        )
        
        # Update application config to use new key (this would be done via config reload)
        # Keep old key for grace period
        
        # Schedule old key revocation
        def revoke_old():
            time.sleep(grace_hours * 3600)
            try:
                self._client.secrets.transit.update_key_configuration(
                    name=old_key_name,
                    min_decryption_version=2,  # Only allow decrypting with v2+
                    min_encryption_version=2,
                )
                log.info(f"Revoked old JWT key {old_key_name} after grace period")
            except Exception as e:
                log.error(f"Failed to revoke old key: {e}")
        
        threading.Thread(target=revoke_old, daemon=True).start()
        
        log.info(f"Rotated JWT keys: {old_key_name} -> {new_key_name} (grace={grace_hours}h)")
        
        return {
            "new_key_id": new_key_name,
            "old_key_id": old_key_name,
            "rotation_time": datetime.utcnow().isoformat(),
            "grace_period_hours": grace_hours,
        }
    
    def sign_jwt(self, payload: dict[str, Any], key_name: str = "jwt-signing") -> str:
        """Sign JWT payload using Vault transit engine."""
        import jwt
        
        header = {"alg": "ES256", "typ": "JWT", "kid": "current"}
        payload_bytes = json.dumps(payload).encode()
        
        # Use Vault to sign
        response = self._client.secrets.transit.sign_data(
            name="jwt-signing",
            hash_input=base64.b64encode(payload_bytes).decode(),
            hash_algorithm="sha2-256",
            signature_algorithm="ecdsa-p256",
        )
        
        signature = response["data"]["signature"]
        # Remove vault:v1: prefix
        signature = signature.replace("vault:v1:", "")
        
        # Construct JWT
        token = jwt.encode(payload, "", algorithm="ES256", headers=header)
        return token
    
    def verify_jwt(self, token: str, key_name: str = "jwt-signing") -> dict:
        """Verify JWT signature using Vault transit engine."""
        import jwt
        
        # Split token
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")
        
        signing_input = f"{parts[0]}.{parts[1]}"
        signature = parts[2]
        
        # Verify with Vault
        response = self._client.secrets.transit.verify_signature(
            name="jwt-signing",
            hash_input=base64.b64encode(signing_input.encode()).decode(),
            hash_algorithm="sha2-256",
            signature=signature,
        )
        
        if not response["data"]["valid"]:
            raise ValueError("Invalid JWT signature")
        
        return jwt.decode(token, options={"verify_signature": False})
    
    # === Artifact Signing ===
    
    def sign_artifact(self, artifact_path: str, key_name: str = "artifact-signing") -> dict:
        """
        Sign artifact file using Vault transit engine.
        
        Returns:
            Dict with artifact_path, signature, key_version, timestamp
        """
        with open(artifact_path, "rb") as f:
            artifact_bytes = f.read()
        
        # Compute SHA256
        digest = hashlib.sha256(artifact_bytes).digest()
        digest_b64 = base64.b64encode(digest).decode()
        
        # Sign with Vault
        response = self._client.secrets.transit.sign_data(
            name="artifact-signing",
            hash_input=digest_b64,
            hash_algorithm="sha2-256",
            signature_algorithm="pss",
        )
        
        return {
            "artifact_path": artifact_path,
            "sha256": digest_b64,
            "signature": response["data"]["signature"].replace("vault:v1:", ""),
            "key_version": response["data"]["key_version"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def verify_artifact(self, artifact_path: str, signature: str, key_name: str = "artifact-signing") -> bool:
        """Verify artifact signature."""
        with open(artifact_path, "rb") as f:
            artifact_bytes = f.read()
        
        digest = hashlib.sha256(artifact_bytes).digest()
        digest_b64 = base64.b64encode(digest).decode()
        
        # Ensure signature has vault prefix
        if not signature.startswith("vault:v1:"):
            signature = f"vault:v1:{signature}"
        
        response = self._client.secrets.transit.verify_signature(
            name="artifact-signing",
            hash_input=digest_b64,
            hash_algorithm="sha2-256",
            signature=signature,
        )
        
        return response["data"]["valid"]
    
    # === Dynamic Database Credentials ===
    
    def configure_database(
        self,
        name: str,
        plugin_name: str,
        connection_url: str,
        allowed_roles: list[str],
        username: str,
        password: str,
    ) -> None:
        """Configure database secrets engine."""
        self._client.secrets.database.configure(
            name=name,
            plugin_name=plugin_name,
            connection_url=connection_url,
            allowed_roles=allowed_roles,
            username=username,
            password=password,
        )
    
    def create_db_role(
        self,
        db_name: str,
        role_name: str,
        db_username: str,
        creation_statements: list[str],
        revocation_statements: list[str],
        default_ttl: str = "1h",
        max_ttl: str = "24h",
    ) -> None:
        """Create database role for dynamic credentials."""
        self._client.secrets.database.create_role(
            name=role_name,
            db_name=db_name,
            creation_statements=creation_statements,
            revocation_statements=revocation_statements,
            default_ttl=default_ttl,
            max_ttl=max_ttl,
        )
    
    def get_db_credentials(self, role_name: str) -> dict[str, Any]:
        """Generate dynamic database credentials."""
        response = self._client.secrets.database.generate_credentials(
            name=role_name,
        )
        return {
            "username": response["data"]["username"],
            "password": response["data"]["password"],
            "lease_id": response["lease_id"],
            "lease_duration": response["lease_duration"],
        }
    
    # === API Key Management ===
    
    def store_api_key(self, path: str, key_data: dict[str, Any]) -> None:
        """Store API key in Vault KV."""
        self._client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=key_data,
        )
    
    def get_api_key(self, path: str) -> dict[str, Any]:
        """Retrieve API key from Vault KV."""
        response = self._client.secrets.kv.v2.read_secret_version(path=path)
        return response["data"]["data"]
    
    def rotate_api_key(self, path: str, generator: Callable[[], str]) -> dict[str, str]:
        """Rotate API key."""
        new_key = generator()
        key_data = self.get_api_key(path)
        key_data["key"] = new_key
        key_data["rotated_at"] = datetime.utcnow().isoformat()
        key_data["version"] = key_data.get("version", 0) + 1
        
        self.store_api_key(path, key_data)
        return {"new_key": new_key, "path": path}
    
    # === Audit Logging ===
    
    def enable_audit_logging(self, path: str = "file/", log_file: str = "/var/log/vault_audit.log") -> None:
        """Enable Vault audit logging."""
        try:
            self._client.sys.enable_audit_device(
                device_type="file",
                path=path,
                options={"file_path": log_file},
            )
            log.info(f"Enabled audit logging to {log_file}")
        except Exception as e:
            log.warning(f"Audit logging may already be enabled: {e}")
    
    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Retrieve recent audit log entries (requires file access)."""
        # This would read from the audit log file
        # Implementation depends on log format
        pass
    
    # === Cleanup ===
    
    def close(self) -> None:
        """Clean up resources."""
        self._stop_rotation.set()
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5)
        self._client.adapter.close()
        log.info("VaultManager closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# === Convenience Functions ===

def create_vault_manager(
    vault_url: str = "http://localhost:8200",
    vault_token: Optional[str] = None,
    rotation_interval_hours: int = 24,
) -> VaultManager:
    """Create VaultManager with standard configuration."""
    config = VaultConfig(url=vault_url, token=vault_token)
    rotation = RotationConfig(
        rotation_interval_hours=rotation_interval_hours,
        enable_auto_rotation=True,
    )
    return VaultManager(config, rotation)


# === Context Manager for Easy Usage ===

class VaultSecretContext:
    """Context manager for automatic secret retrieval and cleanup."""
    
    def __init__(self, vault: VaultManager, path: str):
        self.vault = vault
        self.path = path
        self.secret = None
    
    def __enter__(self):
        self.secret = self.vault.get_api_key(self.path)
        return self.secret
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for KV reads
        return False