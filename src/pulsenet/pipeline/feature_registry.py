"""
Feature Registry — Unified feature engineering for training and inference.
Provides consistent rolling features and column ordering (Staff-level Gap 1).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pulsenet.core.exceptions import DataError
from pulsenet.logger import get_logger

log = get_logger(__name__)


class FeatureRegistry:
    """Centralized feature store logic to eliminate training-serving skew."""

    def __init__(self, rolling_window: int = 5):
        self.rolling_window = rolling_window
        self.feature_cols: list[str] = []
        self.scaler: Optional[MinMaxScaler] = None
        self.is_fitted = False

    def get_feature_names(self, raw_columns: list[str]) -> list[str]:
        """Determine full feature set including generated rolling columns."""
        sensor_cols = [c for c in raw_columns if str(c).startswith("sensor_")]
        rolling_cols = [f"{c}_rolling_mean" for c in sensor_cols]
        return sensor_cols + rolling_cols

    def process_offline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch data for training (Offline)."""
        df = df.copy()
        sensor_cols = [c for c in df.columns if str(c).startswith("sensor_")]
        
        for col in sensor_cols:
            df[f"{col}_rolling_mean"] = df.groupby("unit_number")[col].transform(
                lambda s: s.rolling(window=self.rolling_window, min_periods=1).mean()
            )
        
        # Save feature list for inference consistency
        self.feature_cols = [
            str(c) for c in df.columns 
            if str(c).startswith("sensor_")
        ]
        return df

    def process_online(self, data: dict[str, Any], history: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Process single event for inference (Online)."""
        if not self.is_fitted:
            raise DataError("Feature Registry must be fitted with a scaler first.")

        # Ensure we have a dataframe for rolling calculation
        if history is not None and not history.empty:
            # Append latest reading to history
            current_df = pd.concat([history, pd.DataFrame([data])], ignore_index=True)
            # Compute rolling for only the last row
            for col in self.feature_cols:
                if not col.endswith("_rolling_mean") and f"{col}_rolling_mean" in self.feature_cols:
                    val = current_df[col].tail(self.rolling_window).mean()
                    data[f"{col}_rolling_mean"] = val
        else:
            # Fallback: if no history, rolling mean = current value
            for col in self.feature_cols:
                if not col.endswith("_rolling_mean") and f"{col}_rolling_mean" in self.feature_cols:
                    data[f"{col}_rolling_mean"] = data.get(col, 0.0)

        # Enforce column order
        X = pd.DataFrame([data])[self.feature_cols]
        
        # Scale
        if self.scaler:
            X.loc[:, :] = self.scaler.transform(X)
            
        return X.to_numpy()

    def fit_scaler(self, df: pd.DataFrame, scaler: Optional[MinMaxScaler] = None) -> MinMaxScaler:
        """Fit or set the internal scaler."""
        self.feature_cols = [str(c) for c in df.columns if str(c).startswith("sensor_")]
        X = df[self.feature_cols]
        
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
            
        self.is_fitted = True
        return self.scaler

    def save_config(self) -> dict[str, Any]:
        """Export registry config for persistence."""
        return {
            "rolling_window": self.rolling_window,
            "feature_cols": self.feature_cols,
            "is_fitted": self.is_fitted,
        }

    def load_config(self, config: dict[str, Any]):
        """Load registry config."""
        self.rolling_window = config["rolling_window"]
        self.feature_cols = config["feature_cols"]
        self.is_fitted = config["is_fitted"]
