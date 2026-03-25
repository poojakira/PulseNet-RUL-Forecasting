"""
FastAPI application — central API entry point.
"""

from __future__ import annotations

import time
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
from pulsenet.models.isolation_forest import IsolationForestModel
from pulsenet.security.blockchain import BlackBoxLedger
from pulsenet.pipeline.orchestrator import PipelineOrchestrator
from pulsenet.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model & ledger. Shutdown: cleanup."""
    log.info("PulseNet API starting up")

    registry = ModelRegistry()
    ledger = BlackBoxLedger()
    pipeline = PipelineOrchestrator()

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

    # Wire up dependencies
    set_model_cache({
        "model": model if model_loaded else None,
        "model_name": "isolation_forest",
        "feature_names": feature_names,
    })
    set_pipeline_ref({"pipeline": pipeline})
    set_health_refs({"model": model if model_loaded else None, "registry": registry, "ledger": ledger})
    set_audit_refs({"ledger": ledger})

    yield  # App runs

    log.info("PulseNet API shutting down")


def create_app() -> FastAPI:
    """Factory for the FastAPI application."""
    app = FastAPI(
        title="PulseNet Predictive Maintenance API",
        description="Production-grade anomaly detection for aerospace engine health monitoring",
        version="2.0.0",
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

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "error_code": "INTERNAL_ERROR"},
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

    # Mount routers
    app.include_router(predict.router)
    app.include_router(train.router)
    app.include_router(health.router)
    app.include_router(audit.router)

    return app


app = create_app()
