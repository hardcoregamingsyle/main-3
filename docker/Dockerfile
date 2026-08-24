# ================================================================
# MoE Ultra Engine - Production Dockerfile
# Multi-stage build for optimized image size and security
# ================================================================
# Build arg for Python version (default: 3.11)
ARG PYTHON_VERSION=3.11

# =================================================================
# STAGE 1: Builder Stage
# =================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies into virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only production dependencies
RUN pip install --upgrade pip setuptools wheel && \\
    pip install --no-cache-dir -r requirements.txt

# =================================================================
# STAGE 2: Production Stage
# =================================================================
FROM python:${PYTHON_VERSION}-slim AS production

# Labels for container metadata
LABEL maintainer="Engineering Team <engineering@thalamus.ai>"
LABEL version="1.0.0"
LABEL description="MoE Ultra Engine - Ultra-memory-efficient inference engine"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    APP_HOME=/app \\
    PORT=8000 \\
    HOST=0.0.0.0

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set working directory
WORKDIR $APP_HOME

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup core/ ./core/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup db/ ./db/
COPY --chown=appuser:appgroup pyproject.toml .
COPY --chown=appuser:appgroup README.md .

# Create necessary directories
RUN mkdir -p logs data models && \\
    chown -R appuser:appgroup $APP_HOME

# Switch to non-root user
USER appuser

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command - run the API server
CMD ["uvicorn", "api.main:app", "--host", "${HOST}", "--port", "${PORT}", "--workers", "4"]

# =================================================================
# Alternative: Development Dockerfile (uncomment if needed)
# =================================================================
# FROM python:${PYTHON_VERSION}-slim AS development
# ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# RUN apt-get update && apt-get install -y build-essential curl git && rm -rf /var/lib/apt/lists/*
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
# COPY . .
# USER root
# CMD ["bash"]

# End of Dockerfile