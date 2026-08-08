"""
src/pulsenet/api/metrics.py
──────────────────────────────────────────────────────────────────────────────
Prometheus metrics definitions and /metrics endpoint for PulseNet FastAPI.

All metrics are defined as module-level singletons so they can be imported
and incremented from anywhere in the application without risk of double-
registration.  The /metrics route is designed to be registered on the
internal-only network interface; the docker-compose.yml ensures this endpoint
is NOT exposed on the public-facing port.

Usage (in main.py / app factory):
    from pulsenet.api.metrics import metrics_router
    app.include_router(metrics_router)

Then instrument handlers:
    from pulsenet.api.metrics import (
        PREDICTION_REQUESTS,
        ADVERSARIAL_DETECTIONS,
        INFERENCE_LATENCY,
    )
    PREDICTION_REQUESTS.labels(tenant_id="acme", model="isolation_forest").inc()
    with INFERENCE_LATENCY.labels(model="isolation_forest").time():
        result = model.predict(features)
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ── Counters ───────────────────────────────────────────────────────────────────

PREDICTION_REQUESTS: Counter = Counter(
    name="prediction_requests_total",
    documentation=(
        "Total number of inference requests received by the PulseNet API. "
        "Labeled by tenant_id and model name so per-tenant usage is auditable."
    ),
    labelnames=["tenant_id", "model"],
)

ADVERSARIAL_DETECTIONS: Counter = Counter(
    name="adversarial_detections_total",
    documentation=(
        "Number of requests blocked by the adversarial input guard. "
        "A spike here indicates an active attack or a misconfigured sensor."
    ),
    labelnames=["tenant_id", "detection_reason"],
)

RBAC_VIOLATIONS: Counter = Counter(
    name="rbac_violations_total",
    documentation=(
        "Number of authorization failures. "
        "Labeled by violation_type: expired_token, wrong_tenant, insufficient_role, etc."
    ),
    labelnames=["violation_type"],
)

AUDIT_LOG_EVENTS: Counter = Counter(
    name="audit_log_events_total",
    documentation=(
        "Number of events written to the hash-chained audit log. "
        "Labeled by event_type to distinguish predictions, rbac_violations, etc."
    ),
    labelnames=["event_type"],
)

# ── Histograms ─────────────────────────────────────────────────────────────────

INFERENCE_LATENCY: Histogram = Histogram(
    name="inference_latency_seconds",
    documentation=(
        "Model inference wall-clock time in seconds (excludes network and auth overhead). "
        "Benchmarked baseline: mean=0.0027s, P99=0.0043s on NASA C-MAPSS FD001."
    ),
    labelnames=["model"],
    buckets=(
        0.001,  # 1ms
        0.002,  # 2ms
        0.003,  # 3ms
        0.005,  # 5ms
        0.010,  # 10ms
        0.025,  # 25ms
        0.050,  # 50ms
        0.100,  # 100ms
        0.250,  # 250ms
        0.500,  # 500ms
        1.000,  # 1s
    ),
)

REQUEST_DURATION: Histogram = Histogram(
    name="request_duration_seconds",
    documentation=(
        "Total FastAPI request duration in seconds (includes auth, preprocessing, inference, "
        "and response serialization). Use this for SLO tracking."
    ),
    labelnames=["method", "endpoint", "status_code"],
    buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000),
)

# ── Gauges ─────────────────────────────────────────────────────────────────────

MODEL_DRIFT_SCORE: Gauge = Gauge(
    name="model_drift_score",
    documentation=(
        "Current distribution drift score between live inference inputs and the training "
        "distribution. Computed via KL-divergence on feature histograms. "
        "Alert threshold: 0.15 (see monitoring/alert_rules.yml)."
    ),
    labelnames=["model"],
)

ACTIVE_TENANTS: Gauge = Gauge(
    name="active_tenants",
    documentation=(
        "Number of tenants with at least one valid JWT session in the last 15 minutes. "
        "A sudden drop may indicate a mass-logout or authentication service disruption."
    ),
)

# ── FastAPI router ─────────────────────────────────────────────────────────────

metrics_router = APIRouter(tags=["observability"])


@metrics_router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus metrics scrape endpoint",
    description=(
        "Exposes all registered Prometheus metrics in the Prometheus text exposition format. "
        "This endpoint is restricted to the internal Docker network (pulsenet_internal) and "
        "must NOT be exposed on the public-facing port. "
        "Prometheus scrapes this endpoint every 15 seconds."
    ),
    include_in_schema=False,  # Hide from public OpenAPI docs
)
async def metrics_endpoint() -> Response:
    """Return Prometheus metrics in text exposition format.

    Returns
    -------
    Response
        Plain-text Prometheus metrics payload with content-type
        ``text/plain; version=0.0.4; charset=utf-8``.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ── Request timing middleware ──────────────────────────────────────────────────


def make_metrics_middleware() -> Callable:
    """Return a Starlette-compatible middleware that records REQUEST_DURATION.

    Register in your app factory::

        app.middleware("http")(make_metrics_middleware())

    The middleware records per-route, per-method, per-status-code duration so
    you can build per-endpoint SLO dashboards in Grafana.
    """

    async def middleware(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Use the matched route template as the label to avoid high-cardinality
        # label explosion from path parameters like /predict/{unit_id}.
        route = request.scope.get("path", request.url.path)

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=route,
            status_code=str(response.status_code),
        ).observe(duration)

        return response

    return middleware


# ── Convenience helpers called from route handlers ─────────────────────────────


def record_prediction(tenant_id: str, model: str, latency_seconds: float) -> None:
    """Increment prediction counter and record inference latency.

    Call this once per successful prediction from the route handler.

    Parameters
    ----------
    tenant_id:
        The authenticated tenant making the request.
    model:
        Model name (e.g. ``"isolation_forest"``, ``"lstm_autoencoder"``).
    latency_seconds:
        Wall-clock inference time in seconds.
    """
    PREDICTION_REQUESTS.labels(tenant_id=tenant_id, model=model).inc()
    INFERENCE_LATENCY.labels(model=model).observe(latency_seconds)
    AUDIT_LOG_EVENTS.labels(event_type="prediction_request").inc()


def record_adversarial_detection(tenant_id: str, reason: str) -> None:
    """Increment the adversarial detection counter.

    Parameters
    ----------
    tenant_id:
        Tenant whose request was blocked.
    reason:
        Short reason code (e.g. ``"z_score_exceeded"``, ``"feature_range_violation"``).
    """
    ADVERSARIAL_DETECTIONS.labels(tenant_id=tenant_id, detection_reason=reason).inc()
    AUDIT_LOG_EVENTS.labels(event_type="adversarial_input_blocked").inc()


def record_rbac_violation(violation_type: str) -> None:
    """Increment the RBAC violation counter.

    Parameters
    ----------
    violation_type:
        Short code: ``"expired_token"``, ``"wrong_tenant"``, ``"insufficient_role"``,
        ``"missing_claim"``, etc.
    """
    RBAC_VIOLATIONS.labels(violation_type=violation_type).inc()
    AUDIT_LOG_EVENTS.labels(event_type="rbac_violation").inc()


def update_drift_score(model: str, score: float) -> None:
    """Set the current drift gauge for *model*.

    Parameters
    ----------
    model:
        Model name.
    score:
        KL-divergence drift score (0.0 = no drift; >0.15 triggers alert).
    """
    MODEL_DRIFT_SCORE.labels(model=model).set(score)


def set_active_tenants(count: int) -> None:
    """Update the active-tenants gauge.

    Parameters
    ----------
    count:
        Current number of tenants with active JWT sessions.
    """
    ACTIVE_TENANTS.set(count)
