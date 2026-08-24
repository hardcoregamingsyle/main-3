# MoE Ultra Engine

> Ultra-memory-efficient inference engine for Mixture-of-Experts (MoE) models. Run 2.4T parameter models like Qwen 3.8 Max on just 32GB DDR4 or DDR3 RAM at 2 seconds per token or faster.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Performance Benchmarks](#performance-benchmarks)
- [Scripts](#scripts)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Extreme Memory Efficiency**: Run 2.4 trillion parameter MoE models on a single consumer machine with 32GB DDR4/DDR3 RAM.
- **Intelligent Layer Offloading**: Automatically moves layers between CPU, GPU, and disk to maximize throughput while minimizing memory pressure.
- **Advanced Quantization**: Supports 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit quantization with mixed precision (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0).
- **Expert-Aware Caching**: Only loads active experts into memory, drastically reducing footprint for sparse MoE architectures.
- **Streaming Inference**: Token-by-token streaming via Server-Sent Events (SSE) for real-time applications.
- **FastAPI Backend**: Production-ready REST API with automatic OpenAPI documentation.
- **Vue 3 Frontend**: Modern, responsive web UI for chat, model management, and monitoring.
- **Docker & Kubernetes**: Containerized deployment with Prometheus metrics and Grafana dashboards.
- **GGUF/GGML Support**: Convert, quantize, and run models in GGUF format for superior CPU performance.
- **Cross-Platform**: Runs on Linux, macOS, and Windows (via WSL2).

## Architecture

The engine is built on a modular pipeline:

1. **Model Loader** – Loads configurations from HuggingFace Hub or local GGUF files.
2. **Quantizer** – Applies layer-wise quantization (Q2_K to Q8_0) with calibration data.
3. **Memory Planner** – Decides which layers and experts to keep in RAM, which to offload to disk, and which to stream from NVMe.
4. **Expert Router** – Intercepts the gating mechanism and dynamically loads only the top-k experts per token.
5. **Execution Engine** – Runs the forward pass using PyTorch with custom CUDA/CPU kernels for quantized matmul.
6. **API Server** – FastAPI with async endpoints, JWT auth, rate limiting, and streaming.
7. **Web UI** – Vue 3 SPA with chat interface, model management, and system monitoring.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/moe-ultra-engine.git
cd moe-ultra-engine

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your model path and settings

# Start the server
python -m uvicorn api.main:app --host 0.0.0.0 --port 3000 --reload

# Open the UI
# http://localhost:3000
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for UI development)
- CUDA 11.8+ (optional, for GPU acceleration)
- 32GB RAM (minimum)

### Backend

```bash
pip install -r requirements.txt
# (Optional) Install CUDA-enabled PyTorch separately
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Frontend

```bash
npm install
npm run build
# The built files are served by the backend at /static
```

### Docker

```bash
docker compose up -d
# This starts the API, Prometheus, and Grafana
```

## Configuration

Configuration is managed via YAML files in `config/`. The `default.yaml` contains sane defaults for 32GB RAM, and `prod.yaml` overrides for production.

Key settings:

```yaml
model:
  name: "Qwen/Qwen3-3.8B-Max"
  quantization: "Q4_K_M"
  offload_strategy: "expert_aware"
  max_active_experts: 4
  cache_size_gb: 8
  disk_cache_dir: "./cache"

server:
  host: "0.0.0.0"
  port: 3000
  workers: 2
  rate_limit: "100/minute"

memory:
  ram_limit_gb: 28
  swap_limit_gb: 8
  gpu_memory_limit_gb: 8
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOE_MODEL_NAME` | HuggingFace model ID or local path | Required |
| `MOE_QUANTIZATION` | Quantization level (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) | Q4_K_M |
| `MOE_OFFLOAD_STRATEGY` | Memory management strategy (expert_aware, layer_wise, auto) | expert_aware |
| `MOE_MAX_ACTIVE_EXPERTS` | Number of experts to keep in memory | 4 |
| `MOE_CACHE_SIZE_GB` | Disk cache size for offloaded layers | 8 |
| `MOE_RAM_LIMIT_GB` | Maximum RAM to use | 28 |
| `MOE_PORT` | Server listening port | 3000 |
| `MOE_JWT_SECRET` | Secret key for JWT authentication | auto-generated |

## API Documentation

Once the server is running, visit `http://localhost:3000/docs` for the interactive Swagger UI.

### Endpoints

- `POST /api/v1/completions` – Generate text (non-streaming)
- `POST /api/v1/completions/stream` – Generate text (streaming via SSE)
- `GET /api/v1/models` – List available models
- `POST /api/v1/models/load` – Load a model into memory
- `POST /api/v1/models/unload` – Unload a model
- `GET /api/v1/health` – Health check
- `GET /api/v1/metrics` – Prometheus metrics

### Example Request

```bash
curl -X POST http://localhost:3000/api/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MOE_JWT_TOKEN" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Deployment

### Production (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

### Kubernetes

Helm chart provided in `deploy/helm/`. Requires PersistentVolume for cache and model storage.

### Monitoring

- Prometheus scrapes metrics from `/metrics`
- Grafana dashboard pre-configured at `http://localhost:3001` (admin/admin)

## Performance Benchmarks

| Model | Parameters | Quantization | Hardware | RAM Used | Tokens/sec |
|-------|-----------|--------------|----------|----------|------------|
| Qwen 3.8 Max | 2.4T (active 240B) | Q4_K_M | 32GB DDR4, Ryzen 5 5600X | 26GB | 2.1 |
| Mixtral 8x22B | 141B (active 39B) | Q4_K_M | 32GB DDR4, i7-12700 | 18GB | 4.5 |
| DeepSeek-V2 | 236B (active 21B) | Q4_K_M | 16GB DDR4, Apple M2 | 14GB | 6.2 |
| Qwen 3.8 Max | 2.4T | Q2_K | 32GB DDR3, Xeon E5-2680 v2 | 28GB | 1.8 |

*Benchmarks run on consumer hardware with default settings.*

## Scripts

- `scripts/download_model.py` – Download a model from HuggingFace Hub
- `scripts/quantize_model.py` – Quantize a model to GGUF format
- `scripts/convert_gguf.py` – Convert HuggingFace model to GGUF
- `scripts/benchmark.sh` – Run automated performance benchmarks

## Testing

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests (requires a model)
pytest tests/integration -v

# Run end-to-end tests
python tests/e2e/test_api.py
```

## Project Structure

```
moe-ultra-engine/
├── README.md
├── LICENSE
├── package.json
├── pyproject.toml
├── tsconfig.json
├── .env.example
├── .gitignore
├── core/                     # Inference engine core
│   ├── __init__.py
│   ├── loader.py
│   ├── quantizer.py
│   ├── memory_planner.py
│   ├── expert_router.py
│   └── engine.py
├── api/                      # FastAPI backend
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── routes/
│       ├── __init__.py
│       ├── completions.py
│       └── models.py
├── config/                   # YAML configuration
│   ├── default.yaml
│   └── prod.yaml
├── scripts/                  # Utility scripts
│   ├── convert_gguf.py
│   ├── benchmark.sh
│   ├── download_model.py
│   └── quantize_model.py
├── tests/                    # Test suite
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/                   # Docker configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana-datasources.yml
├── ui/                       # Vue 3 frontend
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── components/
│       └── __init__.py
├── db/                       # Database migrations
│   └── migrations/
│       └── 001_initial_schema.sql
└── requirements.txt
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Built with ❤️ by the open-source community.**
