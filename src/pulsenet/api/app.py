# pyre-ignore-all-errors
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
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pulsenet.api.schemas import TokenRequest, TokenResponse
from pulsenet.api.auth import authenticate_user, create_token
from pulsenet.api.routes import predict, train, health, audit
from pulsenet.api.routes.predict import set_model_cache
from pulsenet.api.routes.train import set_pipeline_ref
from pulsenet.api.routes.health import set_health_refs
from pulsenet.api.routes.audit import set_audit_refs
from pulsenet.models.registry import ModelRegistry
from pulsenet.security.blockchain import BlackBoxLedger
from pulsenet.pipeline.orchestrator import PipelineOrchestrator
from pulsenet.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_event = asyncio.Event()


def _signal_handler(sig, frame):
    """Handle SIGTERM/SIGINT for clean container shutdown."""
    log.info(f"Received signal {sig}. Initiating graceful shutdown...")
    _shutdown_event.set()


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model & ledger. Shutdown: cleanup."""
    log.info("PulseNet API starting up")

    registry = ModelRegistry()
    ledger = BlackBoxLedger()
    pipeline = PipelineOrchestrator()

    from pulsenet.api.routes.predict import batcher

    await batcher.start()
    log.info("Dynamic batcher worker started")

    # Try to load existing model
    model_path = Path("models/isolation_forest.joblib")
    if not model_path.exists():
        model_path = Path("isolation_forest_model.joblib")

    model = registry.get_model("isolation_forest")
    model_loaded = False
    feature_names = []

    if model_path.exists():
        try:
            model.load(model_path)
            model_loaded = True
            if hasattr(model.model, "feature_names_in_"):
                feature_names = list(model.model.feature_names_in_)
            log.info("Model loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load model: {e}")

    # Load scaler
    scaler_path = Path("models/scaler.joblib")
    scaler = None
    if scaler_path.exists():
        import joblib
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
            "feature_names": feature_names,
            "scaler": scaler,
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

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request Correlation ID middleware ---
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """Inject X-Request-ID into every request/response for distributed tracing."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # --- Rate Limiting middleware ---
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
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
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        from pulsenet.api._prometheus import REQUEST_COUNT, REQUEST_LATENCY

        @app.middleware("http")
        async def prometheus_middleware(request: Request, call_next):
            method = request.method
            endpoint = request.url.path
            start = time.time()
            response = await call_next(request)
            duration = time.time() - start
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            return response

        @app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
        async def metrics():
            """Prometheus-compatible metrics endpoint."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        log.info("Prometheus metrics enabled at /metrics")
    except ImportError:
        log.warning("prometheus-client not installed — /metrics endpoint disabled")

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        log.error(f"Unhandled exception: {exc}", extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "error_code": "INTERNAL_ERROR",
                "request_id": request_id,
            },
        )

    # Token endpoint (no auth required)
    @app.post("/token", response_model=TokenResponse, tags=["Authentication"])
    async def login(request: TokenRequest):
        """Authenticate and receive JWT token."""
        user = authenticate_user(request.username, request.password)
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

    # Also keep routes at root for backward compatibility
    app.include_router(predict.router)
    app.include_router(train.router)

    return app


app = create_app()
