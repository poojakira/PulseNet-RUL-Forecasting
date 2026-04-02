# pyre-ignore-all-errors
"""
POST /predict — Real-time inference endpoint.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Union

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from pulsenet.api.auth import require_permission
from pulsenet.api.schemas import (BatchPredictionResponse, BatchSensorInput,
                                  PredictionResponse, SensorInput)
from pulsenet.logger import get_logger
from pulsenet.security.audit import AuditLogger

log = get_logger(__name__)

router = APIRouter(tags=["Inference"])
audit = AuditLogger()

# Module-level model cache (set during app lifespan)
_model_cache: dict = {}


class DynamicBatcher:
    """Groups concurrent FastAPI requests into optimal batches for GPU throughput."""

    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 50):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.timeout_sec = timeout_ms / 1000.0
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self._process_loop())

    async def stop(self):
        if self.task:
            self.task.cancel()

    async def predict_async(
        self, features: Union[list[Any], dict[str, Any]], username: str, role: str
    ) -> PredictionResponse:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((features, username, role, future))
        return await future

    async def _process_loop(self):
        while True:
            try:
                batch = []
                item = await self.queue.get()
                batch.append(item)

                start_time = time.monotonic()
                while len(batch) < self.max_batch_size:
                    elapsed = time.monotonic() - start_time
                    time_left = self.timeout_sec - elapsed
                    if time_left <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(
                            self.queue.get(), timeout=time_left
                        )
                        batch.append(next_item)
                    except asyncio.TimeoutError:
                        break

                await self._run_inference_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Dynamic batcher loop error: {e}")

    async def _run_inference_batch(self, batch):
        model = _model_cache.get("model")
        if not model:
            for _, _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(
                        HTTPException(status_code=503, detail="Model not loaded")
                    )
            return

        model_name = _model_cache.get("model_name", "isolation_forest")
        feature_names = _model_cache.get("feature_names", [])

        features_list = [b[0] for b in batch]
        X = (
            pd.DataFrame(features_list, columns=feature_names)
            if feature_names
            else pd.DataFrame(features_list)
        )

        scaler = _model_cache.get("scaler")

        # If model expects 28 features (raw + rolling) but user sent 14, backfill rolling with raw
        if feature_names and len(X.columns) < len(feature_names):
            sensors = [c for c in X.columns if not c.endswith("_rolling_mean")]
            for s in sensors:
                if (
                    f"{s}_rolling_mean" in feature_names
                    and f"{s}_rolling_mean" not in X.columns
                ):
                    X[f"{s}_rolling_mean"] = X[s]

        # Apply MinMaxScaler if present
        if scaler and hasattr(scaler, "transform"):
            try:
                # Need to enforce column order to match training
                if feature_names:
                    X = X[feature_names]
                X.loc[:, :] = scaler.transform(X)
            except Exception as e:
                log.error(f"Failed to scale input data: {e}")

        t0 = time.perf_counter()
        try:
            # Sync model call wrapped in asyncio.to_thread if it's blocking
            # But since it's HPC, let's execute directly (assuming GPU is fast)
            preds = model.predict(X)
            scores = model.score(X)
            try:
                healths = (
                    model.health_index(X)
                    if hasattr(model, "health_index")
                    else (1 - preds) * 100
                )
            except Exception:
                healths = (1 - preds) * 100

            latency_ms = (time.perf_counter() - t0) * 1000

            for i, (feats, username, role, fut) in enumerate(batch):
                pred = int(preds[i])
                score_val = float(scores[i])
                health = float(healths[i])
                status_str = (
                    "CRITICAL"
                    if pred == 1
                    else ("WARNING" if score_val > -0.02 else "OPTIMAL")
                )

                # Audit
                audit.log_access(
                    endpoint="/predict",
                    method="POST",
                    user=username,
                    role=role,
                    metadata={
                        "dynamic_batch_size": len(batch),
                        "latency_ms": round(latency_ms, 2),
                        "prediction": pred,
                    },
                )

                resp = PredictionResponse(
                    prediction=pred,
                    health_index=round(health, 2),
                    anomaly_score=round(score_val, 6),
                    status=status_str,
                    model_used=model_name,
                )
                if not fut.done():
                    fut.set_result(resp)

        except Exception as e:
            for _, _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)


batcher = DynamicBatcher()


def set_model_cache(cache: dict[str, Any]) -> None:
    """Update the global model cache with a new model and metadata."""
    global _model_cache
    _model_cache = cache


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    data: SensorInput,
    user: dict = Depends(require_permission("predict")),
):
    """Run inference with dynamic batching (groups concurrent requests)."""
    feature_names = _model_cache.get("feature_names", [])
    sensor_dict = data.model_dump()
    values = (
        [sensor_dict.get(f, 0.0) for f in feature_names]
        if feature_names
        else sensor_dict
    )

    return await batcher.predict_async(values, user["username"], user["role"])


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

    X = (
        pd.DataFrame(rows, columns=feature_names)
        if feature_names
        else pd.DataFrame([r.model_dump() for r in data.readings])
    )

    scaler = _model_cache.get("scaler")

    if feature_names and len(X.columns) < len(feature_names):
        sensors = [c for c in X.columns if not c.endswith("_rolling_mean")]
        for s in sensors:
            if (
                f"{s}_rolling_mean" in feature_names
                and f"{s}_rolling_mean" not in X.columns
            ):
                X[f"{s}_rolling_mean"] = X[s]

    if scaler and hasattr(scaler, "transform"):
        try:
            if feature_names:
                X = X[feature_names]
            X.loc[:, :] = scaler.transform(X)
        except Exception as e:
            log.error(f"Failed to scale batch input data: {e}")

    preds = model.predict(X)
    scores = model.score(X)

    try:
        healths = (
            model.health_index(X)
            if hasattr(model, "health_index")
            else (1 - preds) * 100
        )
    except Exception:
        healths = (1 - preds) * 100

    results = []
    for i in range(len(preds)):
        status_str = "CRITICAL" if preds[i] == 1 else "OPTIMAL"
        results.append(
            PredictionResponse(
                prediction=int(preds[i]),
                health_index=round(float(healths[i]), 2),
                anomaly_score=round(float(scores[i]), 6),
                status=status_str,
                model_used=model_name,
            )
        )

    audit.log_access(
        endpoint="/predict/batch",
        method="POST",
        user=user["username"],
        role=user["role"],
        metadata={"batch_size": len(data.readings), "anomalies": int(preds.sum())},
    )

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        anomalies_detected=int(preds.sum()),
        model_used=model_name,
    )
