# pyright: reportGeneralTypeIssues=false
"""
Authenticated Fernet encryption with key rotation support.

Important terminology:
- Fernet provides authenticated symmetric encryption.
- It must not be described as AES-256-GCM.

Key loading order:
  1. Environment variable PULSENET_ENCRYPTION_KEY
  2. Local key file .runtime/pulsenet-fernet.key
  3. Generate a new local development key only when neither exists

Invalid configured keys fail closed. Decryption failures are never converted into
valid-looking sensor values.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

from pulsenet.logger import get_logger

log = get_logger(__name__)


class EncryptionManager:
    """Authenticated Fernet encryption with explicit key handling."""

    def __init__(
        self,
        key_env_var: str = "PULSENET_ENCRYPTION_KEY",
        key_file: str | Path = ".runtime/pulsenet-fernet.key",
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

    def _load_or_generate_key(self) -> bytes:
        """Load a valid key from environment or file, otherwise create a dev key."""
        env_val = os.environ.get(self.key_env_var)
        if env_val:
            key_bytes = env_val.encode()
            try:
                Fernet(key_bytes)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"{self.key_env_var} is set but is not a valid Fernet key"
                ) from exc
            self._key_source = "environment"
            return key_bytes

        if self.key_file.exists():
            key = self.key_file.read_bytes().strip()
            try:
                Fernet(key)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Encryption key file is invalid: {self.key_file}"
                ) from exc
            self._key_source = "file"
            if self._should_rotate(self.key_file):
                log.warning(
                    "Encryption key is due for rotation",
                    extra={"age_days": self._key_age_days(self.key_file)},
                )
            return key

        self._key_source = "generated"
        key = Fernet.generate_key()
        try:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key_file.write_bytes(key)
            try:
                os.chmod(self.key_file, 0o600)
            except OSError:
                log.warning("Could not set key file permissions")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to persist generated encryption key to {self.key_file}"
            ) from exc
        return key

    def rotate_key(self) -> bytes:
        """Rotate the active key and preserve the previous key as a backup.

        Existing ciphertext must be re-encrypted before the backup is removed.
        """
        old_backup = self.key_file.with_suffix(".key.bak")
        if self.key_file.exists():
            if old_backup.exists():
                raise RuntimeError(
                    f"Refusing rotation because backup already exists: {old_backup}"
                )
            self.key_file.rename(old_backup)

        new_key = Fernet.generate_key()
        try:
            self.key_file.write_bytes(new_key)
            os.chmod(self.key_file, 0o600)
        except OSError as exc:
            if old_backup.exists() and not self.key_file.exists():
                old_backup.rename(self.key_file)
            raise RuntimeError("Failed to persist rotated encryption key") from exc

        self._key = new_key
        self._cipher = Fernet(new_key)
        log.info("Key rotated successfully", extra={"backup": str(old_backup)})
        return new_key

    def _should_rotate(self, path: Path) -> bool:
        if not path.exists():
            return False
        return self._key_age_days(path) > self.rotation_days

    @staticmethod
    def _key_age_days(path: Path) -> float:
        try:
            return (time.time() - path.stat().st_mtime) / 86400
        except OSError:
            return 0.0

    def encrypt(self, plaintext: str) -> str:
        return self._cipher.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._cipher.decrypt(ciphertext.encode()).decode()
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise ValueError("Ciphertext authentication or decoding failed") from exc

    def encrypt_bytes(self, data: bytes) -> bytes:
        return self._cipher.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        try:
            return self._cipher.decrypt(data)
        except InvalidToken as exc:
            raise ValueError("Ciphertext authentication failed") from exc

    def encrypt_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info(
            "Encrypting DataFrame", extra={"rows": len(df), "cols": len(df.columns)}
        )
        return pd.DataFrame(
            df.apply(lambda col: col.astype(str).apply(lambda value: self.encrypt(value)))
        )

    def decrypt_dataframe(self, df_enc: pd.DataFrame) -> pd.DataFrame:
        log.info(
            "Decrypting DataFrame",
            extra={"rows": len(df_enc), "cols": len(df_enc.columns)},
        )
        return pd.DataFrame(
            df_enc.apply(
                lambda col: col.astype(str).apply(lambda value: self.decrypt(value))
            )
        )

    def decrypt_cell(self, val: str) -> float:
        """Decrypt one numeric cell.

        Raises ValueError when authentication fails or plaintext is not numeric.
        Returning a default sensor value would hide corruption and is prohibited.
        """
        plaintext = self.decrypt(val)
        try:
            return float(plaintext)
        except (TypeError, ValueError) as exc:
            raise ValueError("Decrypted cell is not a valid numeric value") from exc

    def encrypt_payload(self, payload: dict[str, Any]) -> str:
        return self.encrypt(json.dumps(payload, default=str))

    def decrypt_payload(self, ciphertext: str) -> dict[str, Any]:
        result = json.loads(self.decrypt(ciphertext))
        if not isinstance(result, dict):
            raise ValueError("Decrypted payload is not a dictionary")
        return result
