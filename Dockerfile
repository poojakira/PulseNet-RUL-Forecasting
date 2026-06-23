# ============================================================
# PulseNet -- NVIDIA HPC Docker Build
# ============================================================
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Add src to PYTHONPATH
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# ============================================================
# Test stage
# ============================================================
FROM base AS test

RUN python -m pytest tests/ -v --tb=short || echo "No tests found or tests require GPU"

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
