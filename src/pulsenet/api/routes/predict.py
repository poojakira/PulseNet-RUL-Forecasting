"""
Prediction routes with RBAC protection and input validation.
"""

from __future__ import annotations

import math
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

from pulsenet.api.auth import require_permission

predict_router = APIRouter(prefix="/predict", tags=["Prediction"])


class SensorReading(BaseModel):
    """Single sensor reading with validation."""
    sensor_id: str = Field(..., min_length=1, max_length=50)
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
    engine_id: str = Field(..., min_length=1, max_length=100)
    readings: List[SensorReading] = Field(..., min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_readings(self) -> "PredictionRequest":
        # Check timestamps are strictly increasing
        timestamps = [r.timestamp for r in self.readings]
        if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
            raise ValueError("Timestamps must be strictly increasing")
        return self


class PredictionResponse(BaseModel):
    """Single prediction response."""
    engine_id: str
    rul: float
    anomaly_score: float
    is_anomaly: bool
    timestamp: float
    request_id: str


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    samples: List[PredictionRequest] = Field(..., min_length=1, max_length=100)

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v: List[PredictionRequest]) -> List[PredictionRequest]:
        if len(v) > 100:
            raise ValueError("Maximum 100 samples per batch")
        return v


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    results: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float


# Require 'predict' permission for all prediction endpoints
_predict_deps = [Depends(require_permission("predict"))]


@predict_router.post("", response_model=PredictionResponse, dependencies=_predict_deps)
async def predict(request: PredictionRequest, req: Request) -> PredictionResponse:
    """Single prediction endpoint with RBAC protection."""
    from pulsenet.api.routes.predict import get_model_cache
    import time

    cache = get_model_cache()
    model = cache.get("model")
    registry = cache.get("registry")

    if not model or not registry:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = getattr(req.state, "request_id", "unknown")

    # Transform readings to feature vector
    try:
        features = _transform_readings(request.readings, registry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run prediction
    start = time.time()
    rul, anomaly_score, is_anomaly = model.predict(features)
    processing_time = (time.time() - start) * 1000

    return PredictionResponse(
        engine_id=request.engine_id,
        rul=float(rul),
        anomaly_score=float(anomaly_score),
        is_anomaly=bool(is_anomaly),
        timestamp=time.time(),
        request_id=request_id,
    )


@predict_router.post("/batch", response_model=BatchPredictResponse, dependencies=_predict_deps)
async def predict_batch(request: BatchPredictRequest, req: Request) -> BatchPredictResponse:
    """Batch prediction endpoint with RBAC protection."""
    from pulsenet.api.routes.predict import get_model_cache
    import time

    cache = get_model_cache()
    model = cache.get("model")
    registry = cache.get("registry")

    if not model or not registry:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    results = []

    for sample in request.samples:
        try:
            features = _transform_readings(sample.readings, registry)
            rul, anomaly_score, is_anomaly = model.predict(features)
            results.append(PredictionResponse(
                engine_id=sample.engine_id,
                rul=float(rul),
                anomaly_score=float(anomaly_score),
                is_anomaly=bool(is_anomaly),
                timestamp=time.time(),
                request_id=getattr(req.state, "request_id", "unknown"),
            ))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Sample {sample.engine_id}: {e}")

    return BatchPredictResponse(
        results=results,
        total_processed=len(results),
        processing_time_ms=(time.time() - start) * 1000,
    )


def _transform_readings(readings: List[SensorReading], registry) -> List[float]:
    """Transform validated readings to model feature vector."""
    # Map readings to registry features
    feature_map = {r.sensor_id: r.value for r in readings}

    # Ensure all required features present
    features = []
    for feature_name in registry.feature_names:
        if feature_name not in feature_map:
            raise ValueError(f"Missing required sensor: {feature_name}")
        features.append(feature_map[feature_name])

    return features