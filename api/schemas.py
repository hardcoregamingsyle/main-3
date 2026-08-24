"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    """Request schema for text generation."""
    prompt: str = Field(..., min_length=1, max_length=32768, description="Input prompt")
    max_new_tokens: Optional[int] = Field(default=512, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    top_k: Optional[int] = Field(default=50, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: Optional[bool] = Field(default=True, description="Whether to use sampling")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducibility")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: Optional[bool] = Field(default=False, description="Stream tokens as they are generated")

    @field_validator('stop_sequences')
    @classmethod
def validate_stop_sequences(cls, v):
        if v is not None:
            for seq in v:
                if len(seq) > 100:
                    raise ValueError("Stop sequence too long (max 100 chars)")
        return v


class GenerationResponse(BaseModel):
    """Response schema for text generation."""
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_per_second: float = Field(..., description="Generation speed")
    finish_reason: Literal["stop", "length", "stop_sequence"] = Field(..., description="Why generation stopped")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class StreamToken(BaseModel):
    """Single token in a streaming response."""
    token: str
    token_id: int
    is_final: bool = False
    finish_reason: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information schema."""
    model_id: str
    model_type: str
    parameter_count: int
    expert_count: int
    experts_per_token: int
    hidden_size: int
    num_layers: int
    quantization: str
    dtype: str
    device_map: Dict[str, Any]
    memory_usage: Dict[str, float]


class ModelListResponse(BaseModel):
    """Response for listing available models."""
    models: List[ModelInfo]
    default_model: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    memory: Dict[str, float]
    gpu: Optional[Dict[str, Any]] = None
    model_loaded: bool
    current_model: Optional[str] = None
    queue_size: int = 0


class MetricsResponse(BaseModel):
    """Prometheus-style metrics response."""
    requests_total: int
    requests_active: int
    tokens_generated_total: int
    avg_tokens_per_second: float
    avg_latency_ms: float
    memory_usage_gb: float
    gpu_memory_usage_gb: Optional[float] = None
    offload_events: int
    prefetch_hits: int
    prefetch_misses: int


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    code: str
    timestamp: str


class ConfigUpdateRequest(BaseModel):
    """Request to update runtime configuration."""
    max_gpu_memory_gb: Optional[float] = Field(default=None, ge=0)
    max_cpu_memory_gb: Optional[float] = Field(default=None, ge=0)
    offload_strategy: Optional[str] = None
    experts_on_gpu: Optional[int] = Field(default=None, ge=0)
    enable_prefetch: Optional[bool] = None
    prefetch_layers: Optional[int] = Field(default=None, ge=0)


class BenchmarkRequest(BaseModel):
    """Request schema for benchmarking."""
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=100, ge=1, le=2048)
    num_runs: int = Field(default=3, ge=1, le=10)
    warmup_runs: int = Field(default=1, ge=0, le=5)


class BenchmarkResponse(BaseModel):
    """Benchmark results."""
    avg_tokens_per_second: float
    avg_latency_ms: float
    min_tokens_per_second: float
    max_tokens_per_second: float
    std_dev: float
    runs: List[Dict[str, float]]
