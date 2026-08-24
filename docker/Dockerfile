# Stage 1: Builder for Python Dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md ./
COPY requirements.txt ./ 2>/dev/null || true

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -e .[dev,ml]

# Stage 2: Frontend Build
FROM node:20-alpine AS frontend-builder

WORKDIR /ui

# Copy frontend files
COPY ui/package*.json ./
RUN npm ci

COPY ui/ ./
RUN npm run build

# Stage 3: Runtime Image
FROM python:3.11-slim AS runner

WORKDIR /app

# Labels for OCI standards
LABEL org.opencontainers.image.source="https://github.com/thalamus-project/moe-ultra-engine"
LABEL org.opencontainers.image.description="Ultra-memory-efficient MoE inference engine"
LABEL org.opencontainers.image.version="1.0.0"

# Create non-root user
RUN groupadd -r moeuser && useradd -r -g moeuser moeuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    gosu \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/moeuser/.local
ENV PATH=/home/moeuser/.local/bin:$PATH

# Copy frontend build
COPY --from=frontend-builder /ui/dist /app/ui/dist

# Copy application code
COPY --chown=moeuser:moeuser api/ /app/api/
COPY --chown=moeuser:moeuser core/ /app/core/
COPY --chown=moeuser:moeuser db/ /app/db/
COPY --chown=moeuser:moeuser config/ /app/config/
COPY --chown=moeuser:moeuser scripts/ /app/scripts/
COPY --chown=moeuser:moeuser tests/ /app/tests/
COPY --chown=moeuser:moeuser .env.example /app/.env.example
COPY --chown=moeuser:moeuser README.md /app/README.md

# Set permissions
RUN chown -R moeuser:moeuser /app
USER moeuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers $WORKERS_COUNT"]
