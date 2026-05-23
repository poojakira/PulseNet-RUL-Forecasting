"""Official NASA C-MAPSS FD001 data access.

PulseNet verification and tests use the checked-in NASA archive only. The
loader verifies provenance before extracting any files.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from pulsenet.pipeline.ingestion import load_raw, load_rul

NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
NASA_CMAPSS_LANDING_PAGE = (
    "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
)
NASA_CMAPSS_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"


@dataclass(frozen=True)
class OfficialCmapssFD001:
    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series
    archive_path: Path
    archive_sha256: str
    source_url: str = NASA_CMAPSS_URL
    landing_page: str = NASA_CMAPSS_LANDING_PAGE


def load_official_fd001(
    data_dir: Path | str = Path("data/official"),
    *,
    max_train_rows: int | None = 1000,
    max_test_rows: int | None = 600,
    download: bool = False,
) -> OfficialCmapssFD001:
    """Load FD001 from NASA's C-MAPSS archive after hash verification."""
    root = Path(data_dir)
    archive_path = root / "CMAPSSData.zip"
    if not archive_path.exists():
        if not download:
            raise FileNotFoundError(
                f"{archive_path} missing. Download from {NASA_CMAPSS_URL} "
                "or call load_official_fd001(..., download=True)."
            )
        root.mkdir(parents=True, exist_ok=True)
        _download_nasa_archive(archive_path)

    digest = _sha256(archive_path)
    if digest != NASA_CMAPSS_SHA256:
        raise ValueError(
            "CMAPSSData.zip SHA-256 mismatch: "
            f"expected {NASA_CMAPSS_SHA256}, got {digest}"
        )

    extract_dir = root / "CMAPSSData"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive_path, extract_dir)

    train = load_raw(_find_file(extract_dir, "train_FD001.txt"))
    test = load_raw(_find_file(extract_dir, "test_FD001.txt"))
    rul = load_rul(_find_file(extract_dir, "RUL_FD001.txt"))

    if max_train_rows is not None:
        train = train.head(max_train_rows).copy()
    if max_test_rows is not None:
        test = test.head(max_test_rows).copy()

    return OfficialCmapssFD001(
        train=train,
        test=test,
        rul=rul,
        archive_path=archive_path,
        archive_sha256=digest,
    )


def _download_nasa_archive(destination: Path) -> None:
    parsed = urlparse(NASA_CMAPSS_URL)
    if parsed.scheme != "https" or parsed.netloc != "data.nasa.gov":
        raise ValueError(f"refusing non-NASA HTTPS dataset URL: {NASA_CMAPSS_URL}")
    with urllib.request.urlopen(NASA_CMAPSS_URL, timeout=90) as response:  # noqa: S310
        destination.write_bytes(response.read())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe zip member path: {member.filename}")
        archive.extractall(destination)


def _find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return matches[0]
