# pyright: reportGeneralTypeIssues=false
"""
Integrity-verified deserialization helpers.

Model / scaler artifacts (``.skops``, ``.joblib``) are effectively code when
loaded: an attacker who can replace an artifact on disk gains arbitrary code
execution inside the API process. To defend against this we:

  1. Verify the artifact's SHA-256 against a pinned expected value *before*
     deserializing it, and
  2. Refuse ``skops.io.load(trusted=True)`` (which blindly trusts every type in
     the file). Instead we inspect the untrusted types and only load when they
     fall inside an explicit allow-list.

Expected hashes are resolved from (in order):
  * an environment variable ``PULSENET_<NAME>_SHA256`` (NAME = upper-cased file
    stem, non-alphanumerics -> ``_``), or
  * a sidecar file ``<artifact>.sha256`` containing the hex digest.

If no expected hash is configured the loader fails closed in production
(``PULSENET_ENV=production``) and logs a loud warning elsewhere, so a missing
pin can never silently downgrade to "trust everything".
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable

from pulsenet.logger import get_logger

log = get_logger(__name__)

# skops types we consider safe for scaler / preprocessing artifacts. These are
# pure-data container types with no code-execution side effects on construction.
DEFAULT_TRUSTED_SKOPS_TYPES: tuple[str, ...] = (
    "numpy.dtype",
    "numpy.ndarray",
    "sklearn.preprocessing._data.MinMaxScaler",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.preprocessing._data.RobustScaler",
    "sklearn.preprocessing._data.MaxAbsScaler",
)


def _env_key_for(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).upper()
    return f"PULSENET_{stem}_SHA256"


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_sha256(path: Path) -> str | None:
    """Resolve the pinned expected digest for ``path`` (env var or sidecar)."""
    env_val = os.environ.get(_env_key_for(path))
    if env_val:
        return env_val.strip().lower()

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        # Sidecar may contain "<hash>" or "<hash>  filename"
        return sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    return None


def verify_file_integrity(path: Path) -> None:
    """Verify ``path`` against its pinned SHA-256.

    Raises RuntimeError when the digest does not match. When no pin is
    configured, raises in production and warns loudly otherwise.
    """
    path = Path(path)
    expected = expected_sha256(path)
    if expected is None:
        msg = (
            f"No pinned SHA-256 for artifact {path} "
            f"(set {_env_key_for(path)} or provide {path.name}.sha256)"
        )
        if os.environ.get("PULSENET_ENV") == "production":
            log.critical(msg)
            raise RuntimeError(msg)
        log.warning("%s - proceeding UNVERIFIED (non-production)", msg)
        return

    actual = compute_sha256(path)
    if actual.lower() != expected.lower():
        log.critical(
            "Artifact integrity check FAILED for %s (expected %s, got %s)",
            path,
            expected,
            actual,
        )
        raise RuntimeError(f"Integrity verification failed for {path}")
    log.info("Artifact integrity verified: %s", path)


def safe_skops_load(
    path: Path,
    trusted_types: Iterable[str] = DEFAULT_TRUSTED_SKOPS_TYPES,
) -> Any:
    """Integrity-check then load a ``.skops`` file without blanket trust.

    Unlike ``sio.load(trusted=True)`` this refuses to construct any type that is
    not in ``trusted_types``.
    """
    import skops.io as sio

    path = Path(path)
    verify_file_integrity(path)

    untrusted = sio.get_untrusted_types(file=str(path))
    allowed = set(trusted_types)
    disallowed = [t for t in untrusted if t not in allowed]
    if disallowed:
        log.critical(
            "Refusing to load %s: untrusted types present: %s", path, disallowed
        )
        raise RuntimeError(
            f"Untrusted types in {path.name}: {disallowed}. "
            "Review and add to the allow-list only if provably safe."
        )
    return sio.load(str(path), trusted=list(untrusted))


def safe_joblib_load(path: Path) -> Any:
    """Integrity-check then load a joblib artifact."""
    import joblib

    path = Path(path)
    verify_file_integrity(path)
    return joblib.load(path)
