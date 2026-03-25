"""
GET /health — System health status endpoint.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from pulsenet.api.schemas import HealthResponse

router = APIRouter(tags=["System"])

_start_time = time.time()
_health_refs: dict = {}


def set_health_refs(refs: dict) -> None:
    global _health_refs
    _health_refs = refs


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — no auth required."""
    model_loaded = _health_refs.get("model") is not None
    registry = _health_refs.get("registry")
    ledger = _health_refs.get("ledger")

    models_available = registry.available_models if registry else []
    chain_len = len(ledger.chain) if ledger else 0
    chain_valid = ledger.validate_integrity()[0] if ledger else False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version="2.0.0",
        model_loaded=model_loaded,
        models_available=models_available,
        blockchain_blocks=chain_len,
        blockchain_valid=chain_valid,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
