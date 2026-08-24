# MoE Ultra Engine

> Ultra-memory-efficient inference engine for Mixture-of-Experts (MoE) models. Run 2.4T parameter models like Qwen 3.8 Max on just 32GB DDR4 or DDR3 RAM at 2 seconds per token or faster.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Bit-level quantization** (2-bit, 4-bit, 8-bit) for massive memory savings
- **Expert offloading** — dynamically swap experts between GPU, CPU RAM, and disk
- **Layer-wise prefetching** — stream layers asynchronously to minimize idle time
- **Multi-backend support** — PyTorch, ONNX Runtime, GGUF
- **REST API** with FastAPI — token streaming, chat completions, model management
- **Vue.js dashboard** — real-time memory usage, throughput, and expert activity
- **Docker-ready** — Prometheus metrics, Grafana dashboards, health checks
- **OpenAI-compatible API** — drop-in replacement for existing clients

## Architecture Overview

The engine consists of three main layers:

1. **Core Engine** (`core/`) — Python package implementing the MoE inference pipeline:
   - `ExpertManager` — decides which experts to load/evict based on token routing
   - `Quantizer` — applies bit-level quantization (HQQ, GPTQ, AWQ)
   - `Offloader` — manages memory tiers (GPU, CPU, NVMe) using LRU eviction
   - `LayerExecutor` — orchestrates transformer layers with prefetching

2. **API Server** (`api/`) — FastAPI application providing:
   - `/v1/chat/completions` — streaming chat endpoint
   - `/v1/completions` — text completion
   - `/v1/models` — list loaded models
   - `/admin/stats` — real-time engine metrics
   - `/admin/experts` — expert activity heatmap

3. **Web UI** (`ui/`) — Vue 3 dashboard with:
   - Token streaming with syntax highlighting
   - Memory usage graphs (GPU/CPU/Disk)
   - Expert routing visualization
   - Model switching and configuration

### Data Flow

```
[User Request] → [FastAPI] → [Expert Router] → [Quantized Layer] → [Offloader] → [Output Stream]
                                  ↑
                           [Expert Manager]
                           (GPU/CPU/Disk tier)
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for UI development only)
- 32 GB RAM (minimum), 64 GB recommended
- Optional: NVIDIA GPU with 4+ GB VRAM

### Installation

```bash
git clone https://github.com/yourusername/moe-ultra-engine.git
cd moe-ultra-engine

# Install Python dependencies
pip install -e .

# Install UI dependencies (optional)
npm install
```

### Configuration

Copy the example environment file and adjust settings:

```bash
cp .env.example .env
```

Edit `.env` with your model and memory limits:

```env
MODEL_ID=Qwen/Qwen1.5-MoE-A2.7B
MAX_GPU_MEMORY_GB=4
MAX_CPU_MEMORY_GB=28
MAX_DISK_MEMORY_GB=50
```

### Download a Model

```bash
python scripts/download_model.py --model Qwen/Qwen1.5-MoE-A2.7B
```

Or use a GGUF quantized model:

```bash
python scripts/convert_gguf.py --input /path/to/model --output /path/to/gguf
```

### Start the Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 3000
```

Visit `http://localhost:3000/docs` for interactive API documentation.

### Run the UI (Development)

```bash
npm run dev
```

Then open `http://localhost:5173`.

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | HuggingFace model ID or local path | `Qwen/Qwen1.5-MoE-A2.7B` |
| `HF_TOKEN` | HuggingFace token for gated models | (optional) |
| `MAX_GPU_MEMORY_GB` | Max GPU memory to use (set to 0 for CPU-only) | `4` |
| `MAX_CPU_MEMORY_GB` | Max CPU RAM for offloading | `28` |
| `MAX_DISK_MEMORY_GB` | Max disk space for offloading | `50` |
| `OFFLOAD_STRATEGY` | `auto`, `sequential`, `expert_only` | `auto` |
| `QUANTIZATION_BITS` | Bit-depth for quantization (`2`, `4`, `8`) | `4` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_WORKERS` | Number of Uvicorn workers | `1` |

## API Documentation

### Authentication

Set `API_KEY` in `.env` and pass it in the `Authorization` header: `Bearer your-api-key`.

### Endpoints

#### Chat Completions

```
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "model": "qwen-moe",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "max_tokens": 100,
  "temperature": 0.7
}
```

Response (streaming):
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

#### Text Completions

```
POST /v1/completions
Content-Type: application/json

{
  "model": "qwen-moe",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "stream": false
}
```

#### List Models

```
GET /v1/models
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-moe",
      "object": "model",
      "owned_by": "moe-ultra-engine"
    }
  ]
}
```

#### Admin Statistics

```
GET /admin/stats
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
  "gpu_memory_used_gb": 2.1,
  "cpu_memory_used_gb": 15.3,
  "disk_memory_used_gb": 8.0,
  "active_experts": 12,
  "tokens_per_second": 2.3,
  "queue_depth": 0
}
```

#### Expert Activity

```
GET /admin/experts
Authorization: Bearer YOUR_API_KEY
```

Returns a JSON array of expert usage statistics for visualization.

## Deployment

### Docker Compose (Recommended)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

This starts:
- `moe-engine` — the inference server on port 3000
- `prometheus` — metrics collection on port 9090
- `grafana` — dashboards on port 3001 (admin/admin)

### Kubernetes

See `deploy/` directory for Helm charts and manifests.

### Production Tuning

- Set `API_WORKERS` to number of CPU cores.
- Increase `MAX_DISK_MEMORY_GB` for large models.
- Use a dedicated NVMe drive for the offloading cache.
- Enable monitoring with Prometheus/Grafana.

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# End-to-end tests
pytest tests/e2e

# All tests
pytest
```

### Linting

```bash
# Python
ruff check .

# Frontend
npm run lint
```

### Building the UI

```bash
npm run build
```

Static files are served from `ui/dist` by the FastAPI server.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [AirLLM](https://github.com/lyuchenyang/AirLLM) and the research on memory-efficient MoE inference. Special thanks to the open-source community.
