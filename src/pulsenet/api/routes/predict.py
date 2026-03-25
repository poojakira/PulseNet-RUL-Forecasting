"""
POST /predict — Real-time inference endpoint.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from pulsenet.api.schemas import (
    SensorInput, BatchSensorInput,
    PredictionResponse, BatchPredictionResponse,
)
from pulsenet.api.auth import require_permission
from pulsenet.security.audit import AuditLogger

router = APIRouter(tags=["Inference"])
audit = AuditLogger()

# Module-level model cache (set during app lifespan)
_model_cache: dict = {}


def set_model_cache(cache: dict) -> None:
    global _model_cache
    _model_cache = cache


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    data: SensorInput,
    user: dict = Depends(require_permission("predict")),
):
    """Run inference on a single sensor reading."""
    model = _model_cache.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_name = _model_cache.get("model_name", "isolation_forest")
    feature_names = _model_cache.get("feature_names", [])

    # Build feature vector
    sensor_dict = data.model_dump()
    values = [sensor_dict.get(f, 0.0) for f in feature_names]
    X = pd.DataFrame([values], columns=feature_names) if feature_names else pd.DataFrame([sensor_dict])

    t0 = time.perf_counter()
    pred = model.predict(X)[0]
    score_val = model.score(X)[0]

    # Health index
    try:
        health = float(model.health_index(X)[0]) if hasattr(model, "health_index") else (1 - pred) * 100
    except Exception:
        health = (1 - pred) * 100

    latency_ms = (time.perf_counter() - t0) * 1000

    status_str = "CRITICAL" if pred == 1 else ("WARNING" if score_val > -0.02 else "OPTIMAL")

    # Audit log
    audit.log_access(
        endpoint="/predict", method="POST",
        user=user["username"], role=user["role"],
        metadata={"latency_ms": round(latency_ms, 2), "prediction": int(pred)},
    )

    return PredictionResponse(
        prediction=int(pred),
        health_index=round(health, 2),
        anomaly_score=round(float(score_val), 6),
        status=status_str,
        model_used=model_name,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    data: BatchSensorInput,
    user: dict = Depends(require_permission("predict")),
):
    """Run inference on a batch of sensor readings."""
    model = _model_cache.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_name = data.model_name
    feature_names = _model_cache.get("feature_names", [])

    rows = []
    for reading in data.readings:
        d = reading.model_dump()
        rows.append([d.get(f, 0.0) for f in feature_names])

    X = pd.DataFrame(rows, columns=feature_names) if feature_names else pd.DataFrame(
        [r.model_dump() for r in data.readings]
    )

    preds = model.predict(X)
    scores = model.score(X)

    try:
        healths = model.health_index(X) if hasattr(model, "health_index") else (1 - preds) * 100
    except Exception:
        healths = (1 - preds) * 100

    results = []
    for i in range(len(preds)):
        status_str = "CRITICAL" if preds[i] == 1 else "OPTIMAL"
        results.append(PredictionResponse(
            prediction=int(preds[i]),
            health_index=round(float(healths[i]), 2),
            anomaly_score=round(float(scores[i]), 6),
            status=status_str,
            model_used=model_name,
        ))

    audit.log_access(
        endpoint="/predict/batch", method="POST",
        user=user["username"], role=user["role"],
        metadata={"batch_size": len(data.readings), "anomalies": int(preds.sum())},
    )

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        anomalies_detected=int(preds.sum()),
        model_used=model_name,
    )
