"""
Shared pytest fixtures for PulseNet tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import bcrypt
import numpy as np  # pyre-ignore[21]
import pandas as pd  # pyre-ignore[21]
import pytest  # pyre-ignore[21]

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ["PULSENET_JWT_SECRET"] = "test-secret-key-not-for-production"
os.environ["PULSENET_ENV"] = "testing"


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


os.environ["PULSENET_USERS"] = json.dumps(
    {
        "admin": {"hashed_password": _hash_password("admin123"), "role": "admin"},
        "operator": {"hashed_password": _hash_password("ops123"), "role": "operator"},
    }
)


@pytest.fixture(autouse=True)
def reload_auth_system():
    """Refresh auth users while still using real password hashing."""
    from pulsenet.api import auth

    auth.USER_DB = auth._load_users()
    yield


@pytest.fixture
def official_fd001():
    """Official NASA FD001 subset for test fixtures."""
    from pulsenet.pipeline.official_cmapss import load_official_fd001

    return load_official_fd001(PROJECT_ROOT / "data" / "official", download=False)


@pytest.fixture
def sample_sensor_data(official_fd001) -> pd.DataFrame:
    """Official NASA C-MAPSS sensor data for testing."""
    return official_fd001.test.copy()


@pytest.fixture
def sample_features(sample_sensor_data) -> pd.DataFrame:
    """Sensor data with rolling features added."""
    df = sample_sensor_data.copy()
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = df[col].rolling(window=5, min_periods=1).mean()
    return df


@pytest.fixture
def feature_columns(sample_features) -> list[str]:
    """Return feature column names."""
    return [
        str(c)
        for c in sample_features.columns
        if c not in ("unit_number", "time_in_cycles")
    ]


@pytest.fixture
def sample_X(sample_features, feature_columns) -> np.ndarray:
    """Feature matrix for model testing."""
    return sample_features[feature_columns].values


@pytest.fixture
def sample_y() -> np.ndarray:
    """Binary labels from official FD001 RUL ground truth."""
    from pulsenet.pipeline.official_cmapss import load_official_fd001
    from pulsenet.pipeline.preprocessing import create_labels

    fd001 = load_official_fd001(PROJECT_ROOT / "data" / "official", download=False)
    return create_labels(fd001.test, fd001.rul, failure_threshold=125)


@pytest.fixture
def sample_rul(official_fd001) -> pd.Series:
    """Official NASA FD001 RUL ground truth."""
    return official_fd001.rul


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def cmapss_zip() -> Path:
    """Path to the CMAPSSData.zip archive for tests that need it."""
    archive = PROJECT_ROOT / "data" / "official" / "CMAPSSData.zip"
    if not archive.exists():
        pytest.skip("CMAPSSData.zip not available")
    return archive
