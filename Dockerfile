# =============================================================================
# PulseNet — Production Container Image
# =============================================================================
# Multi-stage build:
#   * builder: installs deps into a virtualenv
#   * runtime: minimal slim image with the venv copied in
#
# Hardening (2026 OWASP/Snyk container baseline):
#   * Non-root runtime user (UID 10001)
#   * Pinned base image digest avoidable via tag pin (python:3.11.9-slim-bookworm)
#   * No build toolchain in final image
#   * No secrets baked in — JWT/encryption keys must be injected at runtime
#   * HEALTHCHECK directive for orchestrators (ECS, K8s, docker-compose)
# =============================================================================

# ---------- Builder stage ----------
FROM python:3.11.9-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /build

# Build deps for any wheels that need compilation (numpy, cryptography backports).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ---------- Runtime stage ----------
FROM python:3.11.9-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PULSENET_ENV=production

# Only what we actually need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (CIS Docker Benchmark 4.1)
RUN groupadd --system --gid 10001 pulsenet \
    && useradd --system --uid 10001 --gid pulsenet --no-create-home --shell /usr/sbin/nologin pulsenet

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source with correct ownership
COPY --chown=pulsenet:pulsenet . /app

# Drop privileges
USER pulsenet:pulsenet

# Liveness probe — orchestrators that don't speak HTTP probes can use this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/healthz || exit 1

EXPOSE 8000 8501

# JWT secret + encryption key MUST be injected at runtime via env vars.
# Never bake secrets into the image.
CMD ["python", "main.py"]
