"""Integrity checks for pickle-backed ML artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path | str) -> Path:
    artifact_path = Path(path)
    return artifact_path.with_name(f"{artifact_path.name}.sha256")


def write_sha256_manifest(path: Path | str) -> Path:
    artifact_path = Path(path)
    manifest = manifest_path(artifact_path)
    manifest.write_text(
        f"{sha256_file(artifact_path)}  {artifact_path.name}\n", encoding="utf-8"
    )
    return manifest


def verify_sha256_manifest(path: Path | str) -> None:
    artifact_path = Path(path)
    manifest = manifest_path(artifact_path)
    if not manifest.exists():
        raise FileNotFoundError(f"missing artifact integrity manifest: {manifest}")

    expected = manifest.read_text(encoding="utf-8").split()[0].strip().lower()
    if len(expected) != 64:
        raise ValueError(f"invalid SHA256 manifest for {artifact_path}")
    actual = sha256_file(artifact_path)
    if actual != expected:
        raise ValueError(f"artifact integrity check failed for {artifact_path}")


def verified_joblib_load(path: Path | str) -> Any:
    """Verify the artifact hash before invoking pickle-backed joblib.load."""
    verify_sha256_manifest(path)
    return joblib.load(path)
