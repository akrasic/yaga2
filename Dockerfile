# syntax=docker/dockerfile:1

# =============================================================================
# Smartbox Anomaly Detection - Production Dockerfile
# =============================================================================

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (faster than pip)
RUN uv pip install --system --no-cache -r pyproject.toml

# =============================================================================
# Production image
# =============================================================================
FROM python:3.12-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 smartbox \
    && useradd --uid 1000 --gid smartbox --shell /bin/bash --create-home smartbox

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=smartbox:smartbox pyproject.toml ./
COPY --chown=smartbox:smartbox config.json ./
COPY --chown=smartbox:smartbox smartbox_anomaly/ ./smartbox_anomaly/
COPY --chown=smartbox:smartbox main.py inference.py ./
COPY --chown=smartbox:smartbox anomaly_models.py anomaly_fingerprinter.py time_aware_anomaly_detection.py vmclient.py ./

# Copy entrypoint script
COPY --chown=smartbox:smartbox docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create directories for persistent data
RUN mkdir -p /app/smartbox_models /app/data /app/logs \
    && chown -R smartbox:smartbox /app

# Install the package
RUN pip install --no-cache-dir -e .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Default config path
ENV CONFIG_PATH=/app/config.json

# Directories for bind mounts (will be created if needed)
# These will be mapped to host directories via docker-compose

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import smartbox_anomaly; print('OK')" || exit 1

# Default entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["scheduler"]
