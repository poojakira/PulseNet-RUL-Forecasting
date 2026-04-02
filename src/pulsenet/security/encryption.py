# pyright: reportGeneralTypeIssues=false
"""
AES-256 Fernet encryption with key rotation and secure key management.

Loads encryption key from:
  1. Environment variable  PULSENET_ENCRYPTION_KEY
  2. Local key file  secret.key
  3. Auto-generates a new key if neither exists
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd
from cryptography.fernet import Fernet

from pulsenet.logger import get_logger

log = get_logger(__name__)


class EncryptionManager:
    """AES-256 Fernet encryption with key rotation support."""

    def __init__(
        self,
        key_env_var: str = "PULSENET_ENCRYPTION_KEY",
        key_file: Union[str, Path] = "secret.key",
        rotation_days: int = 30,
    ):
        self.key_env_var = key_env_var
        self.key_file = Path(key_file)
        self.rotation_days = rotation_days
        self._key_source: str = "unknown"
        self._key: bytes = self._load_or_generate_key()
        self._cipher = Fernet(self._key)
        log.info(
            "EncryptionManager initialized", extra={"key_source": self._key_source}
        )

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------
    def _load_or_generate_key(self) -> bytes:
        """Load key from env → file → generate new."""
        env_val = os.environ.get(self.key_env_var)
        if env_val:
            self._key_source = "environment"
            return env_val.encode()

        if self.key_file.exists():
            self._key_source = "file"
            key = self.key_file.read_bytes().strip()
            if self._should_rotate(self.key_file):
                log.warning(
                    "Encryption key is due for rotation",
                    extra={"age_days": self._key_age_days(self.key_file)},
                )
            return key

        # Generate new key
        self._key_source = "generated"
        key = Fernet.generate_key()
        try:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key_file.write_bytes(key)
            try:
                os.chmod(self.key_file, 0o600)
            except Exception:
                pass  # Best effort on Windows
            log.info(
                "New encryption key generated and saved",
                extra={"file": str(self.key_file)},
            )
        except OSError as e:
            log.error(f"Failed to save generated key to {self.key_file}: {e}")

        return key

    def rotate_key(self) -> bytes:
        """Generate a new key, back up old one, and save."""
        old_backup = self.key_file.with_suffix(".key.bak")
        if self.key_file.exists():
            try:
                self.key_file.rename(old_backup)
            except OSError as e:
                log.warning(f"Failed to create key backup: {e}")

        new_key = Fernet.generate_key()
        self.key_file.write_bytes(new_key)
        try:
            os.chmod(self.key_file, 0o600)
        except Exception:
            pass  # Ignore on Windows or filesystems that don't support chmod
        self._key = new_key
        self._cipher = Fernet(new_key)
        log.info("Key rotated successfully", extra={"backup": str(old_backup)})
        return new_key

    def _should_rotate(self, path: Path) -> bool:
        """Check if the key file is older than the rotation period."""
        if not path.exists():
            return False
        age = self._key_age_days(path)
        return age > self.rotation_days

    @staticmethod
    def _key_age_days(path: Path) -> float:
        """Return age of a file in days."""
        try:
            return (time.time() - path.stat().st_mtime) / 86400
        except OSError:
            return 0.0

    # ------------------------------------------------------------------
    # Encrypt / Decrypt primitives
    # ------------------------------------------------------------------
    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string to a base64-encoded ciphertext."""
        return self._cipher.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext back to a string."""
        return self._cipher.decrypt(ciphertext.encode()).decode()

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        return self._cipher.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        return self._cipher.decrypt(data)

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------
    def encrypt_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encrypt every cell in a DataFrame (string representation)."""
        log.info(
            "Encrypting DataFrame", extra={"rows": len(df), "cols": len(df.columns)}
        )
        return pd.DataFrame(
            df.apply(lambda col: col.astype(str).apply(lambda v: self.encrypt(v)))
        )

    def decrypt_dataframe(self, df_enc: pd.DataFrame) -> pd.DataFrame:
        """Decrypt every cell back to string."""
        log.info(
            "Decrypting DataFrame",
            extra={"rows": len(df_enc), "cols": len(df_enc.columns)},
        )
        return pd.DataFrame(
            df_enc.apply(lambda col: col.astype(str).apply(lambda v: self.decrypt(v)))
        )

    def decrypt_cell(self, val: str) -> float:
        """Decrypt a single cell to float (streaming use-case)."""
        try:
            return float(self.decrypt(val))
        except (ValueError, TypeError, Exception) as e:
            log.debug(f"Cell decryption failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # API payload helpers
    # ------------------------------------------------------------------
    def encrypt_payload(self, payload: dict[str, Any]) -> str:
        """Encrypt a JSON-serializable dict."""
        return self.encrypt(json.dumps(payload, default=str))

    def decrypt_payload(self, ciphertext: str) -> dict[str, Any]:
        """Decrypt back to dict."""
        result = json.loads(self.decrypt(ciphertext))
        if not isinstance(result, dict):
            raise ValueError("Decrypted payload is not a dictionary")
        return result
