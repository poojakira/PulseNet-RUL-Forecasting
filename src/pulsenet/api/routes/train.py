"""
POST /train — Model retraining endpoint.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, Depends

from pulsenet.api.auth import require_permission
from pulsenet.api.schemas import TrainRequest, TrainResponse
from pulsenet.logger import get_logger
from pulsenet.security.audit import AuditLogger

router = APIRouter(tags=["Training"])
audit = AuditLogger()
log = get_logger(__name__)

_pipeline_ref: dict = {}


def set_pipeline_ref(ref: dict) -> None:
    global _pipeline_ref
    _pipeline_ref = ref


def _retrain_task(model_name: str, tune: bool) -> None:
    """Background retraining task."""
    try:
        pipeline = _pipeline_ref.get("pipeline")
        if pipeline is None:
            log.error("Pipeline not available for retraining")
            return
        pipeline.run_ingestion()
        pipeline.run_preprocessing()
        pipeline.run_training(model_name=model_name)
        log.info(f"Retraining complete: {model_name}")
    except Exception as e:
        log.error(f"Retraining failed: {e}")


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("train")),
):
    """Trigger model retraining (runs in background)."""
    audit.log_access(
        endpoint="/train",
        method="POST",
        user=user["username"],
        role=user["role"],
        metadata={"model": request.model_name, "tune": request.tune},
    )

    background_tasks.add_task(_retrain_task, request.model_name, request.tune)

    return TrainResponse(
        model=request.model_name,
        version=time.strftime("%Y%m%d_%H%M%S"),
        train_time_sec=0.0,  # background — not yet known
        samples=0,
        status="training_started",
    )
