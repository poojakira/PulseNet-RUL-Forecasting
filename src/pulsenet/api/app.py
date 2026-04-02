# pyright: reportGeneralTypeIssues=false
"""
FastAPI application — central API entry point.

Production features:
- Request correlation IDs (X-Request-ID)
- Rate limiting middleware
- Prometheus metrics (/metrics)
- Graceful SIGTERM/SIGINT shutdown
- Kubernetes probes (/healthz, /readyz)
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import FrameType
from typing import Any, AsyncGenerator, Optional, Union

import joblib
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pulsenet.api.auth import authenticate_user, create_token
from pulsenet.api.middleware.tenant import TenantMiddleware
from pulsenet.api.routes import audit, health, predict, train
from pulsenet.api.routes.audit import set_audit_refs
from pulsenet.api.routes.health import set_health_refs
from pulsenet.api.routes.predict import set_model_cache
from pulsenet.api.routes.train import set_pipeline_ref
from pulsenet.api.schemas import TokenRequest, TokenResponse
from pulsenet.config import cfg
from pulsenet.logger import get_logger
from pulsenet.models.registry import ModelRegistry
from pulsenet.pipeline.feature_registry import FeatureRegistry
from pulsenet.pipeline.orchestrator import PipelineOrchestrator
from pulsenet.security.blockchain import BlackBoxLedger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_event = asyncio.Event()


def _signal_handler(sig: int, frame: Optional[FrameType]) -> None:
    """Handle SIGTERM/SIGINT for clean container shutdown."""
    log.info(f"Received signal {sig}. Initiating graceful shutdown...")
    _shutdown_event.set()


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load model & ledger. Shutdown: cleanup."""
    log.info("PulseNet API starting up")

    registry = ModelRegistry()
    ledger = BlackBoxLedger()
    pipeline = PipelineOrchestrator()
    feature_registry = FeatureRegistry(rolling_window=cfg.data.rolling_window)

    from pulsenet.api.routes.predict import batcher

    await batcher.start()
    log.info("Dynamic batcher worker started")

    # Try to load existing model
    model_path = Path("models/isolation_forest.joblib")
    if not model_path.exists():
        model_path = Path("isolation_forest_model.joblib")

    model = registry.get_model("isolation_forest")
    model_loaded = False
    feature_names: list[str] = []

    if model_path.exists():
        try:
            model.load(model_path)
            model_loaded = True
            if hasattr(model, "model") and hasattr(model.model, "feature_names_in_"):  # type: ignore
                feature_names = list(model.model.feature_names_in_)  # type: ignore
            log.info("Model loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load model: {e}")

    # Load scaler
    scaler_path = Path("models/scaler.joblib")
    scaler: Any = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            log.info("MinMaxScaler loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load scaler: {e}")

    # Wire up dependencies
    set_model_cache(
        {
            "model": model if model_loaded else None,
            "model_name": "isolation_forest",
            "registry": feature_registry,
            "scaler": scaler,
            "ledger": ledger,
            # For Gap 2 (Shadow Mode), let's pre-load the LSTM if it exists as shadow
            "shadow_model": None, # For now, can be populated if lstm.joblib exists
            "shadow_model_name": "lstm"
        }
    )
    set_pipeline_ref({"pipeline": pipeline})
    set_health_refs(
        {
            "model": model if model_loaded else None,
            "registry": registry,
            "ledger": ledger,
        }
    )
    set_audit_refs({"ledger": ledger})

    yield  # App runs

    await batcher.stop()
    log.info("PulseNet API shutting down")


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-IP)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window
        hits = self._requests.get(client_ip, [])
        hits = [t for t in hits if t > window_start]
        hits.append(now)
        self._requests[client_ip] = hits
        return len(hits) <= self.max_requests


_rate_limiter = _RateLimiter(max_requests=100, window_seconds=60)


def create_app() -> FastAPI:
    """Factory for the FastAPI application."""
    app = FastAPI(
        title="PulseNet Predictive Maintenance API",
        description="Production-grade anomaly detection for aerospace engine health monitoring",
        version="2.1.0",
        lifespan=lifespan,
    )

    # CORS Hardware Security: Pull from Config (No wildcards in production)
    origins = cfg.api.cors_origins
    is_production = os.environ.get("PULSENET_ENV") == "production"
    
    if not origins:
        log.warning("No CORS_ORIGINS configured! Defaulting to empty list.")
        origins = []
    elif "*" in origins and is_production:
        log.critical("SECURITY VIOLATION: Wildcard CORS '*' detected in production!")
        # Fallback to a safe empty list if misconfigured in production
        origins = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # --- Multi-Tenancy middleware ---
    app.add_middleware(TenantMiddleware)

    # --- Request Correlation ID middleware ---
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
        """Inject X-Request-ID into every request/response for distributed tracing."""
        request_id = str(request.headers.get("X-Request-ID", uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # --- Rate Limiting middleware ---
    @app.middleware("http")
    async def rate_limit_middleware(
        request: Request, call_next: Any
    ) -> Union[Response, JSONResponse]:
        """Enforce per-IP rate limits."""
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            log.warning("Rate limit exceeded", extra={"client_ip": client_ip})
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                    "error_code": "RATE_LIMITED",
                },
            )
        return await call_next(request)

    # --- Prometheus Metrics ---
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        from pulsenet.api._prometheus import REQUEST_COUNT, REQUEST_LATENCY

        @app.middleware("http")
        async def prometheus_middleware(request: Request, call_next: Any) -> Response:
            method = request.method
            endpoint = request.url.path
            start = time.time()
            response: Response = await call_next(request)
            duration = time.time() - start
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            return response

        @app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
        async def metrics() -> Response:
            """Prometheus-compatible metrics endpoint."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        log.info("Prometheus metrics enabled at /metrics")
    except ImportError:
        log.warning("prometheus-client not installed — /metrics endpoint disabled")

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        log.error(f"Unhandled exception: {exc}", extra={"request_id": request_id}, exc_info=True)
        
        # In production, we mask the actual exception message to avoid leaking internals
        detail = "An internal server error occurred."
        if getattr(cfg.system, "debug", False):
            detail = str(exc)
            
        return JSONResponse(
            status_code=500,
            content={
                "detail": detail,
                "error_code": "INTERNAL_ERROR",
                "request_id": request_id,
            },
        )

    # Token endpoint (no auth required)
    @app.post("/token", response_model=TokenResponse, tags=["Authentication"])
    async def login(request_data: TokenRequest) -> Union[TokenResponse, JSONResponse]:
        """Authenticate and receive JWT token."""
        user = authenticate_user(request_data.username, request_data.password)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid credentials", "error_code": "AUTH_FAILED"},
            )
        token, expiry = create_token(user["username"], user["role"])
        return TokenResponse(
            access_token=token,
            role=user["role"],
            expires_in_minutes=expiry,
        )

    # Mount routers with API version prefix
    api_v1 = FastAPI(title="PulseNet API v1")
    api_v1.include_router(predict.router)
    api_v1.include_router(train.router)

    # Health/audit stay at root level (K8s probes expect /healthz at root)
    app.include_router(health.router)
    app.include_router(audit.router)

    # Mount versioned API
    app.mount("/api/v1", api_v1)

    return app


app = create_app()
