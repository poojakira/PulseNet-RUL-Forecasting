"""
Shared pytest fixtures for PulseNet tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np  # pyre-ignore[21]
import pandas as pd  # pyre-ignore[21]
import pytest  # pyre-ignore[21]

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test environment configuration. Set BEFORE importing pulsenet.api.auth so
# the module-level _JWT_SECRET pickup uses these values.
os.environ.setdefault("PULSENET_JWT_SECRET", "test-secret-key-not-for-production")
os.environ.setdefault("PULSENET_ENV", "testing")


def _build_user_db() -> str:
    """Hash test passwords with REAL bcrypt and return JSON for PULSENET_USERS.

    Previous version monkeypatched _hash_password / _verify_password — that
    bypassed bcrypt entirely, so a bcrypt regression would not be detected
    by the test suite (audit finding). Now we hash test passwords with the
    actual bcrypt implementation so tests exercise the real auth path.
    """
    from passlib.hash import bcrypt as _bcrypt

    users = {
        "admin": {
            "hashed_password": _bcrypt.hash("admin123"),
            "role": "admin",
        },
        "operator": {
            "hashed_password": _bcrypt.hash("ops123"),
            "role": "operator",
        },
    }
    return json.dumps(users)


# Build users with real bcrypt hashes BEFORE pulsenet.api.auth is imported
os.environ.setdefault("PULSENET_USERS", _build_user_db())


@pytest.fixture(autouse=True)
def reload_user_db():
    """Refresh USER_DB before every test so env-var changes take effect."""
    from pulsenet.api import auth

    auth.USER_DB = auth._load_users()
    yield


@pytest.fixture
def sample_sensor_data() -> pd.DataFrame:
    """Generate synthetic sensor data for testing."""
    np.random.seed(42)
    n = 200
    data = {
        "unit_number": np.repeat([1, 2], n // 2),
        "time_in_cycles": np.tile(np.arange(1, n // 2 + 1), 2),
    }
    # Add sensor columns (matching FD001 after dropping noisy ones)
    for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
        data[f"sensor_{s}"] = np.random.randn(n) * 0.1 + 0.5
    return pd.DataFrame(data)


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
    """Binary labels for testing."""
    np.random.seed(42)
    return np.array([0] * 170 + [1] * 30)


@pytest.fixture
def sample_rul() -> pd.Series:
    """Simulated RUL ground truth."""
    return pd.Series([80, 50])


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path
