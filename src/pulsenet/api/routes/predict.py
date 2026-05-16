"""
POST /predict — Real-time inference endpoint with dynamic request batching.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from pulsenet.api.auth import require_permission
from pulsenet.api.schemas import (
    BatchPredictionResponse,
    BatchSensorInput,
    PredictionResponse,
    SensorInput,
)
from pulsenet.logger import get_logger
from pulsenet.pipeline.feature_registry import FeatureRegistry
from pulsenet.security.audit import AuditLogger
from pulsenet.security.blockchain import BlackBoxLedger

log = get_logger(__name__)

router = APIRouter(tags=["Inference"])
audit = AuditLogger()

# Module-level model cache (set during app lifespan)
_model_cache: dict = {}


class DynamicBatcher:
    """Groups concurrent FastAPI requests into batches for GPU throughput optimization."""

    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 50):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.timeout_sec = timeout_ms / 1000.0
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def predict_async(
        self,
        features: Union[list[Any], dict[str, Any]],
        username: str,
        role: str,
        tenant_id: str,
    ) -> PredictionResponse:
        """Submit a request to the batcher and await result."""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((features, username, role, tenant_id, future))
        return await future

    async def _process_loop(self) -> None:
        """Continuously drain the queue, batch requests, run inference."""
        while True:
            try:
                batch: list[tuple] = []
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

    async def _run_inference_batch(self, batch: list[tuple]) -> None:
        """Run model inference on a batch of pending requests."""
        model = _model_cache.get("model")
        ledger: Optional[BlackBoxLedger] = _model_cache.get("ledger")

        if not model:
            for _features, _user, _role, _tenant, fut in batch:
                if not fut.done():
                    fut.set_exception(
                        HTTPException(status_code=503, detail="Model not loaded")
                    )
            return

        model_name = _model_cache.get("model_name", "isolation_forest")
        registry: Optional[FeatureRegistry] = _model_cache.get("registry")
        shadow_model = _model_cache.get("shadow_model")
        shadow_model_name = _model_cache.get("shadow_model_name", "none")

        features_list = [b[0] for b in batch]

        # Unified feature processing via FeatureRegistry (eliminates training-serving skew)
        try:
            if registry and registry.is_fitted:
                X_list = []
                for feat in features_list:
                    X_list.append(registry.process_online(feat).flatten())
                X = pd.DataFrame(X_list, columns=registry.feature_cols)
            else:
                X = pd.DataFrame(features_list)
        except Exception as e:
            log.error(f"Feature processing failed: {e}")
            for _features, _user, _role, _tenant, fut in batch:
                if not fut.done():
                    fut.set_exception(
                        HTTPException(status_code=400, detail=f"Invalid features: {e}")
                    )
            return

        t0 = time.perf_counter()
        try:
            # Run blocking model.predict() in a thread to avoid blocking the event loop
            preds = await asyncio.to_thread(model.predict, X)
            scores = await asyncio.to_thread(model.score, X)

            # Shadow model inference (safe A/B testing pattern)
            shadow_preds = None
            if shadow_model:
                try:
                    shadow_preds = await asyncio.to_thread(shadow_model.predict, X)
                except Exception as e:
                    log.warning(f"Shadow model inference failed: {e}")

            # Compute health index (with fallback)
            try:
                if hasattr(model, "health_index"):
                    healths = await asyncio.to_thread(model.health_index, X)
                else:
                    healths = (1 - preds) * 100
            except Exception:
                healths = (1 - preds) * 100

            latency_ms = (time.perf_counter() - t0) * 1000

            for i, (_feats, username, role, tenant_id, fut) in enumerate(batch):
                pred = int(preds[i])
                score_val = float(scores[i])
                health = float(healths[i])
                status_str = (
                    "CRITICAL"
                    if pred == 1
                    else ("WARNING" if score_val > -0.02 else "OPTIMAL")
                )

                audit_meta: dict[str, Any] = {
                    "dynamic_batch_size": len(batch),
                    "latency_ms": round(latency_ms, 2),
                    "prediction": pred,
                }
                if shadow_preds is not None:
                    s_pred = int(shadow_preds[i])
                    audit_meta["shadow_prediction"] = s_pred
                    audit_meta["model_disagreement"] = bool(pred != s_pred)
                    if pred != s_pred:
                        log.info(
                            "Model Disagreement",
                            extra={
                                "primary_model": model_name,
                                "primary_pred": pred,
                                "shadow_model": shadow_model_name,
                                "shadow_pred": s_pred,
                            },
                        )

                # Tenant-aware audit logging
                audit.log_access(
                    endpoint="/predict",
                    method="POST",
                    user=username,
                    role=role,
                    metadata=audit_meta,
                    tenant_id=tenant_id,
                )

                # Tenant-aware ledger entry (only on critical or model disagreement)
                if ledger and (pred == 1 or audit_meta.get("model_disagreement")):
                    ledger.add_entry(
                        unit_id=-1,  # unit ID not in stateless prediction payload
                        cycles=-1,
                        health_score=health,
                        status=status_str,
                        tenant_id=tenant_id,
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
            log.error(f"Batch inference failed: {e}", exc_info=True)
            # Correctly unpack 5-tuple — was previously a bug
            for _features, _user, _role, _tenant, fut in batch:
                if not fut.done():
                    fut.set_exception(e)


batcher = DynamicBatcher()


def set_model_cache(cache: dict[str, Any]) -> None:
    """Update the global model cache with a new model and metadata."""
    global _model_cache
    _model_cache = cache


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    data: SensorInput,
    user: dict = Depends(require_permission("predict")),
):
    """Run inference with dynamic batching (groups concurrent requests)."""
    sensor_dict = data.model_dump()
    tenant_id = getattr(request.state, "tenant_id", "public")
    return await batcher.predict_async(
        sensor_dict, user["username"], user["role"], tenant_id
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

    X = (
        pd.DataFrame(rows, columns=feature_names)
        if feature_names
        else pd.DataFrame([r.model_dump() for r in data.readings])
    )

    scaler = _model_cache.get("scaler")

    # Add rolling mean columns if missing (training-serving alignment)
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
            log.error(f"Failed to scale batch input: {e}")
            raise HTTPException(
                status_code=400, detail=f"Feature scaling failed: {e}"
            )

    preds = await asyncio.to_thread(model.predict, X)
    scores = await asyncio.to_thread(model.score, X)

    try:
        if hasattr(model, "health_index"):
            healths = await asyncio.to_thread(model.health_index, X)
        else:
            healths = (1 - preds) * 100
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
        metadata={
            "batch_size": len(data.readings),
            "anomalies": int(preds.sum()),
        },
    )

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        anomalies_detected=int(preds.sum()),
        model_used=model_name,
    )
