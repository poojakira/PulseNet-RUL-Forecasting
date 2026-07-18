"""Request/Response schemas with strict validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import math


class SensorReading(BaseModel):
    """Single sensor reading with validation."""
    sensor_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    timestamp: float = Field(..., gt=0, lt=1e12)
    value: float = Field(...)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Sensor value must be finite (not NaN/inf)")
        if abs(v) > 1e6:
            raise ValueError(f"Sensor value out of range: {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        if v <= 0 or v > 1e12:
            raise ValueError("Invalid timestamp")
        return v


class PredictionRequest(BaseModel):
    """Prediction request with validated sensor data."""
    engine_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    readings: List[SensorReading] = Field(..., min_length=1, max_length=1000)

    @field_validator("readings")
    @classmethod
    def validate_readings(cls, v: List[SensorReading]) -> List[SensorReading]:
        timestamps = [r.timestamp for r in v]
        if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
            raise ValueError("Timestamps must be strictly increasing")
        return v


class PredictionResponse(BaseModel):
    """Single prediction response."""
    engine_id: str
    rul: float = Field(..., ge=0)
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomaly: bool
    timestamp: float
    request_id: str


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    samples: List[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    results: List[PredictionResponse]
    total_processed: int = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0)


class TrainRequest(BaseModel):
    """Training request."""
    dataset: str = Field(default="FD001", pattern=r"^FD00[1-4]$")
    retrain: bool = False


class TrainResponse(BaseModel):
    """Training response."""
    status: str
    model_version: str
    training_time_seconds: float
    metrics: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float


class AuditEntry(BaseModel):
    """Audit log entry."""
    timestamp: float
    event_type: str
    user: Optional[str] = None
    details: dict
    request_id: Optional[str] = None


class TokenRequest(BaseModel):
    """Token request."""
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    role: str
    expires_in_minutes: int = Field(..., gt=0)