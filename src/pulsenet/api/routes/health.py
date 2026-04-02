"""
Health endpoints — /health (full), /healthz (liveness), /readyz (readiness).

Includes GPU telemetry via pynvml when available.
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter

from pulsenet.api.schemas import HealthResponse
from pulsenet.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["System"])

_start_time = time.time()
_health_refs: dict = {}


def set_health_refs(refs: dict) -> None:
    global _health_refs
    _health_refs = refs


def _get_gpu_info() -> list[dict]:
    """Collect GPU health via pynvml (NVIDIA Management Library)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = -1
            try:
                power = round(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0, 1)
            except Exception:
                power = -1
            gpus.append(
                {
                    "id": i,
                    "name": str(pynvml.nvmlDeviceGetName(handle)),
                    "utilization_pct": float(util.gpu),
                    "memory_used_mb": round(float(mem.used) / 1024**2),
                    "memory_total_mb": round(float(mem.total) / 1024**2),
                    "temperature_c": temp,
                    "power_watts": power,
                }
            )
        pynvml.nvmlShutdown()
        return gpus
    except ImportError:
        return []
    except Exception as e:
        log.debug(f"GPU info unavailable: {e}")
        return []


def _get_system_resources() -> dict:
    """Collect CPU and memory usage."""
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "cpu_percent": float(proc.cpu_percent(interval=0.1)),
            "memory_rss_mb": round(float(mem.rss) / 1024**2, 1),
            "threads": int(proc.num_threads()),
        }
    except ImportError:
        return {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Full system health — model, GPU, blockchain, resources."""
    model_loaded = _health_refs.get("model") is not None
    registry = _health_refs.get("registry")
    ledger = _health_refs.get("ledger")

    models_available = registry.available_models if registry else []
    chain_len = len(ledger.chain) if ledger else 0
    chain_valid = ledger.validate_integrity()[0] if ledger else False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version="2.1.0",
        model_loaded=model_loaded,
        models_available=models_available,
        blockchain_blocks=chain_len,
        blockchain_valid=chain_valid,
        uptime_seconds=round(time.time() - _start_time, 1),
        gpu_devices=_get_gpu_info(),
        system_resources=_get_system_resources(),
    )


@router.get("/healthz", tags=["Kubernetes"])
async def liveness():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "alive"}


@router.get("/readyz", tags=["Kubernetes"])
async def readiness():
    """Readiness probe — returns 200 only if model is loaded and ready to serve."""
    model_loaded = _health_refs.get("model") is not None
    if not model_loaded:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "model_not_loaded"},
        )
    return {"status": "ready", "model_loaded": True}
