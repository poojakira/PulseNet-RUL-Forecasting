# ============================================================
# PulseNet — NVIDIA HPC Docker Build
# ============================================================
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS base

WORKDIR /app

# System dependencies (skip heavy ML ones since NGC has them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Add src to PYTHONPATH
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
# JWT secret MUST be provided at runtime via environment variable.
# Never bake secrets into the image.
# ENV PULSENET_JWT_SECRET must be set via docker-compose, ECS task def, or K8s secret.

# Expose ports (API: 8000, Dashboard: 8501)
EXPOSE 8000 8501

# Default: start API
CMD ["python", "main.py"]
