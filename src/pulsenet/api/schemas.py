"""
Pydantic schemas for API request/response validation.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ===== Request Models =====


class SensorInput(BaseModel):
    """Single sensor reading for prediction."""

    sensor_2: float = Field(..., description="Sensor 2 reading")
    sensor_3: float = Field(..., description="Sensor 3 reading")
    sensor_4: float = Field(..., description="Sensor 4 reading")
    sensor_7: float = Field(..., description="Sensor 7 reading")
    sensor_8: float = Field(..., description="Sensor 8 reading")
    sensor_9: float = Field(..., description="Sensor 9 reading")
    sensor_11: float = Field(..., description="Sensor 11 reading")
    sensor_12: float = Field(..., description="Sensor 12 reading")
    sensor_13: float = Field(..., description="Sensor 13 reading")
    sensor_14: float = Field(..., description="Sensor 14 reading")
    sensor_15: float = Field(..., description="Sensor 15 reading")
    sensor_17: float = Field(..., description="Sensor 17 reading")
    sensor_20: float = Field(..., description="Sensor 20 reading")
    sensor_21: float = Field(..., description="Sensor 21 reading")


class BatchSensorInput(BaseModel):
    """Batch of sensor readings."""

    readings: list[SensorInput]
    model_name: str = Field(default="isolation_forest", description="Model to use")


class TrainRequest(BaseModel):
    """Training configuration request."""

    model_name: str = Field(default="isolation_forest")
    tune: bool = Field(default=False, description="Run hyperparameter tuning")
    epochs: Optional[int] = Field(
        default=None, description="Training epochs (LSTM/Transformer)"
    )


class TokenRequest(BaseModel):
    """JWT token request."""

    username: str
    password: str


# ===== Response Models =====


class PredictionResponse(BaseModel):
    """Single prediction result."""

    prediction: int = Field(..., description="0=Normal, 1=Anomaly")
    health_index: float = Field(..., description="Health score 0-100%")
    anomaly_score: float = Field(..., description="Raw anomaly score")
    status: str = Field(..., description="OPTIMAL/WARNING/CRITICAL")
    model_used: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]
    total: int
    anomalies_detected: int
    model_used: str


class HealthResponse(BaseModel):
    """System health status."""

    status: str = Field(..., description="healthy/degraded/unhealthy")
    version: str
    model_loaded: bool
    models_available: list[str]
    blockchain_blocks: int
    blockchain_valid: bool
    uptime_seconds: float
    gpu_devices: list[dict[str, Any]] = Field(
        default_factory=list, description="GPU telemetry via pynvml"
    )
    system_resources: dict[str, Any] = Field(
        default_factory=dict, description="CPU/memory metrics"
    )


class TrainResponse(BaseModel):
    """Training result."""

    model: str
    version: str
    train_time_sec: float
    samples: int
    metrics: Optional[dict[str, float]] = None
    status: str


class AuditResponse(BaseModel):
    """Blockchain audit entry."""

    chain_length: int
    merkle_root: Optional[str] = None
    is_valid: bool
    validation_message: str
    recent_blocks: list[dict[str, Any]]


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str
