# pyright: reportGeneralTypeIssues=false
"""
Encryption utilities for PulseNet.

Two independent encryption layers are provided:

1. **Module-level AES-256-GCM functions** (``encrypt`` / ``decrypt``)
   Low-level, key-explicit primitives using AESGCM from the ``cryptography``
   library.  These are the canonical functions for tamper-evident encryption
   of individual payloads.  ``decrypt`` ALWAYS raises ``InvalidTag`` on any
   authentication failure — it never returns a default value.

2. **EncryptionManager class** (Fernet / AES-128-CBC + HMAC-SHA256)
   Higher-level key-management wrapper used for DataFrame and API-payload
   encryption.  Handles key loading, rotation, and file-permission hardening.

Loads encryption key from:
  1. Environment variable  PULSENET_ENCRYPTION_KEY
  2. Local key file  .runtime/pulsenet-fernet.key
  3. Auto-generates a new key if neither exists
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pulsenet.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level AES-256-GCM primitives
# ---------------------------------------------------------------------------
# These functions are intentionally key-explicit (no hidden global state).
# The caller is responsible for generating and protecting the 32-byte key.
#
# Token format:  nonce (12 bytes) || ciphertext+tag (variable)
# The GCM authentication tag is appended to the ciphertext by AESGCM.encrypt.


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt *plaintext* with AES-256-GCM.

    Parameters
    ----------
    key:
        32-byte (256-bit) symmetric key.  Must be generated with a CSPRNG
        (e.g. ``os.urandom(32)``).
    plaintext:
        Arbitrary byte string to protect.

    Returns
    -------
    bytes
        ``nonce || ciphertext || gcm_tag`` — a self-contained token that
        can be passed to :func:`decrypt`.

    Raises
    ------
    ValueError
        If *key* is not exactly 32 bytes.
    """
    if len(key) != 32:
        raise ValueError(
            f"AES-256 requires a 32-byte key; got {len(key)} bytes"
        )
    nonce = os.urandom(12)  # 96-bit nonce; NIST SP 800-38D recommended size
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext_and_tag


def decrypt(key: bytes, token: bytes) -> bytes:
    """Decrypt a token produced by :func:`encrypt`.

    This function is **fail-closed**: it ALWAYS raises ``InvalidTag`` when the
    ciphertext has been tampered with, truncated, or encrypted under a different
    key.  It NEVER returns a default or fallback value.

    Parameters
    ----------
    key:
        32-byte (256-bit) symmetric key identical to the one used to encrypt.
    token:
        Byte string in the format ``nonce || ciphertext || gcm_tag`` as
        produced by :func:`encrypt`.

    Returns
    -------
    bytes
        The original plaintext.

    Raises
    ------
    ValueError
        If *key* is not exactly 32 bytes, or *token* is too short to contain
        a valid nonce.
    cryptography.exceptions.InvalidTag
        If the GCM authentication tag does not match — meaning the ciphertext
        was tampered with, truncated, or decrypted with the wrong key.
        **This exception must never be caught and swallowed by callers.**
    """
    if len(key) != 32:
        raise ValueError(
            f"AES-256 requires a 32-byte key; got {len(key)} bytes"
        )
    if len(token) < 12:
        raise ValueError(
            f"Token too short to contain a 12-byte nonce; got {len(token)} bytes"
        )
    nonce, ciphertext_and_tag = token[:12], token[12:]
    aesgcm = AESGCM(key)
    # AESGCM.decrypt raises InvalidTag on any authentication failure.
    # Do NOT catch InvalidTag here — let it propagate to the caller.
    return aesgcm.decrypt(nonce, ciphertext_and_tag, None)

log = get_logger(__name__)


class EncryptionManager:
    """AES-256 Fernet encryption with key rotation support."""

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

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------
    def _load_or_generate_key(self) -> bytes:
        """Load key from env → file → generate new."""
        env_val = os.environ.get(self.key_env_var)
        if env_val:
            self._key_source = "environment"
            # Validate the key format - must be 32 url-safe base64-encoded bytes
            try:
                key_bytes = env_val.encode()
                Fernet(key_bytes)  # Validate
                return key_bytes
            except Exception:
                log.warning(
                    "PULSENET_ENCRYPTION_KEY is not a valid Fernet key, "
                    "generating new one"
                )
                # Fall through to generate

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
                log.warning("Could not set key file permissions (non-fatal on Windows)")
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
            log.warning(
                "Could not set key file permissions during rotation (non-fatal)"
            )
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
        """Decrypt a single encrypted cell and convert to float (streaming use-case).

        Raises
        ------
        cryptography.fernet.InvalidToken
            If the ciphertext is invalid or has been tampered with.  This
            exception is **intentionally not caught** — a tampered cell must
            never silently produce a synthetic sensor value (e.g. 0.0) that
            could corrupt downstream ICS decisions.
        ValueError
            If the decrypted value cannot be converted to float.
        """
        # Do NOT catch cryptography.fernet.InvalidToken here.  A tampered
        # ciphertext must propagate immediately; returning 0.0 (or any other
        # sentinel) would allow adversarial sensor data to reach the model.
        decrypted_str = self.decrypt(val)
        return float(decrypted_str)

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
