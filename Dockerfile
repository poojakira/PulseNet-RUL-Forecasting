# ============================================================
# PulseNet -- Container Build
# ------------------------------------------------------------
# CPU-only base image. The original NVIDIA NGC PyTorch image
# (nvcr.io/nvidia/pytorch) requires a GPU host and a ~9GB pull
# that is not available on hosted CI runners. A slim CPU base
# keeps the image buildable in CI; requirements.txt installs
# the CPU build of torch. For GPU/HPC deployments, swap the
# base back to nvcr.io/nvidia/pytorch:<tag>.
# ============================================================
FROM python:3.12-slim AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# System dependencies (curl for healthchecks, libgomp for numpy/torch/sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Add src to PYTHONPATH
ENV PYTHONPATH="/app/src"

# ============================================================
# Test stage
# ============================================================
FROM base AS test

RUN python -m pytest tests/ -v --tb=short

# ============================================================
# Runtime stage
# ============================================================
FROM base AS runtime

# Create non-root user
RUN useradd -m -r -s /bin/bash pulsenet && chown -R pulsenet:pulsenet /app
USER pulsenet

# Expose ports (API: 8000, Dashboard: 8501)
EXPOSE 8000 8501

# Default: start API
CMD ["python", "main.py"]
