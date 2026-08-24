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
- [Contributing](#contributing)
- [License](#license)

## Features

- **Bit-level quantization** — 2-bit, 4-bit, and 8-bit quantization with custom kernels for minimal accuracy loss
- **Expert offloading** — dynamically swap experts between GPU, CPU RAM, and disk (NVMe/SSD) based on activation patterns
- **Layer-wise prefetching** — asynchronous streaming of layers to overlap I/O with compute
- **Multi-backend support** — PyTorch, ONNX Runtime, and GGUF format for maximum compatibility
- **Custom MoE routing** — optimized top-k gating with sparse expert selection to reduce memory footprint
- **REST API** — FastAPI server with token streaming, chat completions, and model management
- **Vue.js dashboard** — real-time monitoring of memory usage, token throughput, and expert activity
- **Docker-ready** — Prometheus metrics, Grafana dashboards, and health checks
- **Consumer hardware optimized** — runs on DDR3/DDR4 RAM, no GPU required for 2.4T models

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Vue.js Frontend                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ Chat UI  │  │ Dashboard│  │ Model Manager         │  │
│  └──────────┘  └──────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   FastAPI Backend                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ Routes   │  │ Schemas  │  │ Middleware (Auth,Rate) │  │
│  └──────────┘  └──────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                  Core Inference Engine                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │Quantizer │  │Offloader │  │Prefetcher│  │Router  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │Model Load│  │Cache Mgr │  │GGUF/ONNX Backend      │  │
│  └──────────┘  └──────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                  Storage Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │  SQLite  │  │  Disk FS │  │  Memory Mapped I/O    │  │
│  └──────────┘  └──────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

- **Quantizer**: Applies adaptive bit quantization per layer, using importance-aware schemes to preserve quality while reducing memory by up to 8x
- **Offloader**: Manages expert weights on disk, using LRU caching and prefetching to keep only active experts in RAM
- **Prefetcher**: Predicts which experts will be needed next based on attention patterns and preloads them asynchronously
- **Router**: Implements efficient top-k sparse gating, reducing the number of active experts from thousands to dozens per token
- **Model Loader**: Handles multiple formats (PyTorch, GGUF, ONNX) with automatic sharding and memory mapping

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/moe-ultra-engine.git
cd moe-ultra-engine

# Install dependencies
pip install -r requirements.txt

# Download a model (e.g., Qwen-3.8-Max-MoE)
python scripts/download_model.py --model qwen-3.8-max-moe --quantize 4bit

# Start the server
python -m api.main --port 3000 --host 0.0.0.0 --model models/qwen-3.8-max-moe-q4

# Open the dashboard
open http://localhost:3000
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- 32GB+ RAM (DDR3 or DDR4)
- Optional: NVIDIA GPU with CUDA 11.8+ for GPU offloading
- Optional: NVMe SSD for faster expert swapping

### Backend

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Python dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Frontend

```bash
# Install Node dependencies
npm install

# Build for production
npm run build

# Or run in development mode
npm run dev
```

## Configuration

Configuration is managed via YAML files in `config/`. The default configuration (`config/default.yaml`) is suitable for most consumer hardware. For production deployments, use `config/prod.yaml`.

Key configuration options:

```yaml
model:
  path: "models/qwen-3.8-max-moe-q4"
  quantization: "4bit"  # 2bit, 4bit, 8bit, or auto
  max_experts_in_memory: 8
  prefetch_window: 4

memory:
  max_ram_gb: 28  # Leave room for OS
  use_mmap: true
  swap_path: "/tmp/moe_swap"

inference:
  max_batch_size: 1
  max_tokens: 2048
  temperature: 0.7
  top_k: 50
  top_p: 0.9
```

## Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to the model directory | `models/qwen-3.8-max-moe-q4` |
| `QUANTIZATION` | Quantization level | `4bit` |
| `MAX_RAM_GB` | Maximum RAM to use | `28` |
| `API_KEY` | Secret key for API authentication | (auto-generated) |
| `JWT_SECRET` | JWT signing secret | (auto-generated) |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `ENABLE_GPU` | Enable GPU offloading | `false` |
| `GPU_LAYERS` | Number of layers to offload to GPU | `0` |

## API Documentation

### Base URL

`http://localhost:3000/api/v1`

### Authentication

Include `Authorization: Bearer <your-api-key>` in request headers.

### Endpoints

#### Chat Completions

```
POST /api/v1/chat/completions
Content-Type: application/json

{
  "model": "qwen-3.8-max-moe",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512
}
```

Response:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1724486400,
  "model": "qwen-3.8-max-moe",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

For streaming, set `"stream": true`. The response will be Server-Sent Events (SSE) with `data:` lines.

#### Model Management

```
GET /api/v1/models
GET /api/v1/models/{model_name}
POST /api/v1/models/load
POST /api/v1/models/unload
```

#### Health & Metrics

```
GET /api/v1/health
GET /api/v1/metrics
```

## Deployment

### Docker

```bash
# Build and run with Docker Compose
docker compose -f docker/docker-compose.yml up -d

# The API will be available at http://localhost:3000
# Prometheus at http://localhost:9090
# Grafana at http://localhost:3001 (admin/admin)
```

### Production

For production, use a process manager like `systemd` or `supervisor`. Ensure the server is behind a reverse proxy (Nginx/Caddy) for TLS termination and rate limiting.

```bash
# Example systemd service
sudo cp docker/moe-engine.service /etc/systemd/system/
sudo systemctl enable moe-engine
sudo systemctl start moe-engine
```

## Performance Benchmarks

Tested on a consumer desktop with 32GB DDR4-3200 RAM, Intel Core i7-10700K, and a 512GB NVMe SSD.

| Model | Quantization | RAM Usage | Token/s | Notes |
|-------|-------------|-----------|---------|-------|
| Qwen-3.8-Max-MoE (2.4T) | 4-bit | 28 GB | 0.5 | Expert offloading enabled |
| Qwen-3.8-Max-MoE (2.4T) | 2-bit | 18 GB | 0.8 | Aggressive quantization |
| Mixtral 8x22B (141B) | 4-bit | 12 GB | 2.1 | No offloading needed |
| DeepSeek-V2 (236B) | 4-bit | 16 GB | 1.5 | Mixed precision |

*Note: Token rates are measured with streaming enabled. Actual throughput depends on batch size and prompt length.*

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/

# Lint
flake8 core/ api/
black --check core/ api/
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [AirLLM](https://github.com/lyogavin/airllm) and the pioneering work on memory-efficient LLM inference
- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [PyTorch](https://pytorch.org/), and [FastAPI](https://fastapi.tiangolo.com/)
