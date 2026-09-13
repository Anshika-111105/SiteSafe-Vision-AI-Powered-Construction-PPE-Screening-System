FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Production Runtime Stage
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install curl for healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-privileged system user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy installed python site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code and trained model weights
COPY --chown=appuser:appgroup app /app/app
COPY --chown=appuser:appgroup src /app/src
COPY --chown=appuser:appgroup configs /app/configs
COPY --chown=appuser:appgroup artifacts/models/production_model.pt /app/artifacts/models/production_model.pt
COPY --chown=appuser:appgroup artifacts/models/model_metadata.json /app/artifacts/models/model_metadata.json
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml

# Switch to non-root user
USER appuser:appgroup

# Expose microservice HTTP port
EXPOSE 8000

# Container Healthcheck Probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production startup entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
