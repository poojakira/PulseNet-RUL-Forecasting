# ============================================================
# PulseNet — Multi-stage Docker build
# ============================================================
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Add src to PYTHONPATH
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PULSENET_JWT_SECRET="change-me-in-production"

# Expose ports (API: 8000, Dashboard: 8501)
EXPOSE 8000 8501

# Default: start API
CMD ["python", "main.py"]
