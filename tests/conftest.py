"""
Shared pytest fixtures for PulseNet tests.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np  # pyre-ignore[21]
import pandas as pd  # pyre-ignore[21]
import pytest  # pyre-ignore[21]

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ["PULSENET_JWT_SECRET"] = "test-secret-key"


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
    return [str(c) for c in sample_features.columns
            if c not in ("unit_number", "time_in_cycles")]


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
