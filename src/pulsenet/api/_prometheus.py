"""
Module-level Prometheus metric objects — created once to avoid duplicate registration.
"""

try:
    from prometheus_client import Counter, Histogram

    REQUEST_COUNT = Counter(
        "pulsenet_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "pulsenet_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )
except ImportError:
    REQUEST_COUNT = None  # type: ignore[assignment]
    REQUEST_LATENCY = None  # type: ignore[assignment]
