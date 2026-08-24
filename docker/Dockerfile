# MoE Ultra Engine - Production Dockerfile
# Multi-stage build for minimal image size

# ==========================================
# Build Stage
# ==========================================
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    git \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    bitsandbytes>=0.43.0 \
    auto-gptq>=0.7.0 \
    awq>=0.2.0 \
    flash-attn>=2.5.0 \
    xformers>=0.0.27 \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.6.0 \
    pydantic-settings>=2.2.0 \
    python-dotenv>=1.0.0 \
    prometheus-client>=0.19.0 \
    psutil>=5.9.0 \
    tqdm>=4.66.0 \
    requests>=2.31.0 \
    huggingface-hub>=0.22.0

# ==========================================
# Runtime Stage
# ==========================================
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r moe && useradd -r -g moe -d /app -s /bin/bash moe

# Create app directory
WORKDIR /app

# Copy application code
COPY --chown=moe:moe . /app/

# Create cache and log directories
RUN mkdir -p /var/cache/moe-ultra-engine/offload \
    /var/log/moe-engine \
    /tmp/moe-ultra-engine && \
    chown -R moe:moe /var/cache/moe-ultra-engine /var/log/moe-engine /tmp/moe-ultra-engine

# Switch to non-root user
USER moe

# Environment variables
ENV HF_HOME=/var/cache/moe-ultra-engine/hf
ENV TRANSFORMERS_CACHE=/var/cache/moe-ultra-engine/hf
ENV MOE_OFFLOAD_DIR=/var/cache/moe-ultra-engine/offload
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 3000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Entry point
ENTRYPOINT ["python", "-m", "api.main"]
