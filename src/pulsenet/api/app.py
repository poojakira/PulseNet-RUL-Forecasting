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
import hashlib
import hmac
import json
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import FrameType
from typing import Any, AsyncGenerator, Optional, Union

import joblib
import skops.io as sio
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
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.models.registry import ModelRegistry
from pulsenet.pipeline.feature_registry import FeatureRegistry
from pulsenet.pipeline.orchestrator import PipelineOrchestrator
from pulsenet.security.blockchain import BlackBoxLedger

log = get_logger(__name__)

MODEL_CANDIDATES = (
    Path("models/isolation_forest.skops"),
    Path("models/isolation_forest.joblib"),
    Path("isolation_forest_model.skops"),
)
SCALER_CANDIDATES = (Path("models/scaler.skops"), Path("models/scaler.joblib"))
REGISTRY_CANDIDATES = (Path("models/feature_registry.joblib"),)
ARTIFACT_MANIFEST_CANDIDATES = (Path("models/api_artifacts.sha256.json"),)
ARTIFACT_MANIFEST_KEY_ENV = "PULSENET_ARTIFACT_MANIFEST_KEY"


def _first_existing_path(candidates: tuple[Path, ...], kind: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        f"Critical error: {kind} file not found. Expected one of: {expected}"
    )


def _artifact_manifest_key(path: Path) -> str:
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_signature_payload(payload: dict[str, Any]) -> bytes:
    signed_payload = {
        "schema_version": payload.get("schema_version"),
        "artifacts": payload.get("artifacts"),
    }
    return json.dumps(signed_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _artifact_manifest_key_bytes(signing_key: str | None = None) -> bytes:
    key = signing_key or os.environ.get(ARTIFACT_MANIFEST_KEY_ENV, "")
    if not key:
        raise RuntimeError(
            f"Critical error: {ARTIFACT_MANIFEST_KEY_ENV} is not configured"
        )
    return key.encode("utf-8")


def _verify_artifact_manifest_signature(
    payload: dict[str, Any], signing_key: str | None = None
) -> None:
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise RuntimeError("Critical error: artifact manifest missing signature")

    algorithm = signature.get("algorithm")
    value = signature.get("value")
    if algorithm != "hmac-sha256" or not isinstance(value, str):
        raise RuntimeError("Critical error: invalid artifact manifest signature")

    expected = hmac.new(
        _artifact_manifest_key_bytes(signing_key),
        _artifact_signature_payload(payload),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, value):
        raise RuntimeError("Critical error: artifact manifest signature mismatch")


def _verify_artifact_manifest(
    artifact_paths: tuple[Path, ...],
    manifest_path: Path | None = None,
    signing_key: str | None = None,
) -> Path:
    manifest = manifest_path or _first_existing_path(
        ARTIFACT_MANIFEST_CANDIDATES, "Artifact manifest"
    )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Critical error: invalid artifact manifest {manifest}"
        ) from exc

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Critical error: artifact manifest missing artifacts map")

    _verify_artifact_manifest_signature(payload, signing_key=signing_key)

    for artifact_path in artifact_paths:
        key = _artifact_manifest_key(artifact_path)
        entry = artifacts.get(key)
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else entry
        if not isinstance(expected_hash, str):
            raise RuntimeError(f"Critical error: manifest missing hash for {key}")
        actual_hash = _sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Critical error: artifact hash mismatch for {key}")
    return manifest


def _load_scaler_artifact(scaler_path: Path) -> Any:
    if scaler_path.suffix == ".skops":
        untrusted_types = sio.get_untrusted_types(file=scaler_path)
        if untrusted_types:
            joined = ", ".join(sorted(untrusted_types))
            raise RuntimeError(
                "Critical error: scaler artifact contains untrusted skops types: "
                f"{joined}"
            )
        return sio.load(file=scaler_path, trusted=[])

    if scaler_path.suffix == ".joblib":
        ## Joblib is only accepted for artifacts generated by the local pipeline.
        return joblib.load(scaler_path)

    raise RuntimeError(f"Critical error: unsupported scaler artifact: {scaler_path}")


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

    model_path = _first_existing_path(MODEL_CANDIDATES, "Model")
    scaler_path = _first_existing_path(SCALER_CANDIDATES, "Scaler")
    registry_config_path = _first_existing_path(REGISTRY_CANDIDATES, "Feature registry")
    manifest_path = _verify_artifact_manifest(
        (model_path, scaler_path, registry_config_path)
    )
    log.info("API artifacts verified", extra={"manifest_path": str(manifest_path)})
    model = registry.get_model("isolation_forest")
    model_loaded = False
    feature_names: list[str] = []

    try:
        ## Joblib model artifacts are accepted only from the local generation script.
        if not isinstance(model, IsolationForestModel):
            raise RuntimeError("Registry returned an unexpected model type")
        model.load(model_path, trusted=model_path.suffix != ".skops")
        model_loaded = True
        if hasattr(model, "model") and hasattr(model.model, "feature_names_in_"):  # type: ignore
            feature_names = list(model.model.feature_names_in_)  # type: ignore
        log.info("Model loaded successfully", extra={"model_path": str(model_path)})
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Critical error: Failed to load model {model_path}") from e

    try:
        scaler = _load_scaler_artifact(scaler_path)
        log.info("Scaler loaded successfully", extra={"scaler_path": str(scaler_path)})
    except Exception as e:
        log.error(f"Failed to load scaler: {e}")
        raise RuntimeError(
            f"Critical error: Failed to load scaler {scaler_path}"
        ) from e

    try:
        feature_registry.load_config(joblib.load(registry_config_path))
        feature_registry.scaler = scaler
        feature_registry.is_fitted = True
        log.info(
            "Feature registry loaded successfully",
            extra={"registry_path": str(registry_config_path)},
        )
    except Exception as e:
        log.error(f"Failed to load feature registry: {e}")
        raise RuntimeError(
            f"Critical error: Failed to load feature registry {registry_config_path}"
        ) from e

    # Wire up dependencies
    set_model_cache(
        {
            "model": model if model_loaded else None,
            "model_name": "isolation_forest",
            "feature_names": feature_names,
            "registry": feature_registry,
            "scaler": scaler,
            "ledger": ledger,
            # For Gap 2 (Shadow Mode), let's pre-load the LSTM if it exists as shadow
            "shadow_model": None,  # For now, can be populated if lstm.skops exists
            "shadow_model_name": "lstm",
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
        log.error(
            f"Unhandled exception: {exc}",
            extra={"request_id": request_id},
            exc_info=True,
        )

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
