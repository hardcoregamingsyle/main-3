"""
Configuration management for MoE Ultra Engine.

Uses Pydantic Settings for type-safe configuration with support for:
- YAML config files (default.yaml, prod.yaml)
- Environment variables
- CLI overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model-specific configuration."""
    name: str = Field(default="qwen-3.8-max", description="Model identifier")
    path: str = Field(default="./models", description="Path to model files")
    max_seq_len: int = Field(default=8192, ge=512, le=32768, description="Maximum sequence length")
    dtype: str = Field(default="float16", pattern="^(float16|bfloat16|float32|int8|int4)$")
    quantization: str = Field(default="int4", pattern="^(none|int8|int4|gptq|awq)$")
    expert_parallelism: int = Field(default=1, ge=1, le=8, description="Number of GPUs for expert parallelism")
    tensor_parallelism: int = Field(default=1, ge=1, le=8, description="Number of GPUs for tensor parallelism")
    pipeline_parallelism: int = Field(default=1, ge=1, le=8, description="Number of GPUs for pipeline parallelism")
    rope_theta: float = Field(default=1000000.0, description="RoPE theta parameter")
    rope_scaling: Optional[Dict[str, Any]] = Field(default=None, description="RoPE scaling configuration")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        path = Path(v).expanduser().resolve()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return str(path)


class MemoryConfig(BaseModel):
    """Memory management configuration."""
    max_ram_gb: float = Field(default=32.0, gt=0, le=512, description="Maximum RAM to use (GB)")
    max_vram_gb: float = Field(default=0.0, ge=0, le=512, description="Maximum VRAM to use (GB), 0 = auto")
    offload_to_cpu: bool = Field(default=True, description="Offload inactive experts to CPU")
    offload_to_disk: bool = Field(default=False, description="Offload to disk when RAM full")
    disk_offload_path: str = Field(default="./offload", description="Path for disk offloading")
    expert_cache_size: int = Field(default=4, ge=1, le=64, description="Number of experts to keep in RAM")
    kv_cache_dtype: str = Field(default="float16", pattern="^(float16|bfloat16|float32|int8)$")
    kv_cache_quantization: str = Field(default="none", pattern="^(none|int8|int4)$")
    page_size: int = Field(default=16, ge=1, le=256, description="Paged attention page size")
    max_pages: int = Field(default=0, ge=0, description="Max pages for KV cache, 0 = auto")

    @field_validator("disk_offload_path")
    @classmethod
    def validate_disk_path(cls, v: str) -> str:
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class InferenceConfig(BaseModel):
    """Inference runtime configuration."""
    batch_size: int = Field(default=1, ge=1, le=256, description="Batch size for inference")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    top_k: int = Field(default=50, ge=1, le=1000, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    stream: bool = Field(default=True, description="Stream tokens as they're generated")
    use_flash_attention: bool = Field(default=True, description="Use flash attention if available")
    use_paged_attention: bool = Field(default=True, description="Use paged attention for KV cache")
    speculative_decoding: bool = Field(default=False, description="Enable speculative decoding")
    draft_model_path: Optional[str] = Field(default=None, description="Path to draft model for speculative decoding")


class ServerConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=3000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of worker processes")
    timeout: int = Field(default=300, ge=1, le=3600, description="Request timeout (seconds)")
    max_request_size: int = Field(default=100_000_000, description="Max request size in bytes")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "OPTIONS"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed headers")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window (seconds)")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    access_log: bool = Field(default=True, description="Enable access logging")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    prometheus_port: int = Field(default=9090, ge=1, le=65535, description="Prometheus exporter port")
    grafana_port: int = Field(default=3001, ge=1, le=65535, description="Grafana port")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    jaeger_endpoint: Optional[str] = Field(default=None, description="Jaeger collector endpoint")
    log_format: str = Field(default="json", pattern="^(json|text)$")
    log_file: Optional[str] = Field(default="./logs/moe-engine.log", description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation interval")
    log_retention: str = Field(default="30 days", description="Log retention period")

    @field_validator("log_file")
    @classmethod
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        if v:
            path = Path(v).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        return v


class SecurityConfig(BaseModel):
    """Security configuration."""
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_keys: List[str] = Field(default_factory=list, description="Valid API keys")
    jwt_secret: Optional[str] = Field(default=None, description="JWT secret for token auth")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, ge=1, le=168, description="JWT expiry in hours")
    tls_enabled: bool = Field(default=False, description="Enable TLS")
    tls_cert_path: Optional[str] = Field(default=None, description="TLS certificate path")
    tls_key_path: Optional[str] = Field(default=None, description="TLS key path")


class Config(BaseSettings):
    """Main configuration class combining all sub-configurations."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Runtime fields (not from config files)
    config_file: Optional[str] = Field(default=None, exclude=True)
    profile: str = Field(default="default", exclude=True)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "Config":
        """Validate cross-field constraints."""
        if self.memory.max_ram_gb < 8:
            raise ValueError("max_ram_gb must be at least 8GB for MoE models")
        if self.model.expert_parallelism * self.model.tensor_parallelism * self.model.pipeline_parallelism > 8:
            raise ValueError("Total parallelism (expert * tensor * pipeline) cannot exceed 8")
        if self.security.api_key_enabled and not self.security.api_keys and not self.security.jwt_secret:
            raise ValueError("API key enabled but no API keys or JWT secret configured")
        if self.security.tls_enabled and (not self.security.tls_cert_path or not self.security.tls_key_path):
            raise ValueError("TLS enabled but cert/key paths not configured")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self.model_dump(exclude={"config_file", "profile"})

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@lru_cache(maxsize=1)
def load_config(
    config_file: Optional[Union[str, Path]] = None,
    profile: str = "default",
    env_overrides: bool = True,
) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_file: Path to YAML config file. If None, searches default locations.
        profile: Configuration profile ("default" or "prod").
        env_overrides: Whether to apply environment variable overrides.

    Returns:
        Config: Loaded and validated configuration.
    """
    config_data: Dict[str, Any] = {}

    # Determine config file path
    if config_file is None:
        search_paths = [
            Path.cwd() / f"config/{profile}.yaml",
            Path.cwd() / "config/default.yaml",
            Path(__file__).parent.parent / f"config/{profile}.yaml",
            Path(__file__).parent.parent / "config/default.yaml",
        ]
        for p in search_paths:
            if p.exists():
                config_file = p
                break

    # Load from YAML if found
    if config_file and Path(config_file).exists():
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Create config instance (env vars automatically applied by BaseSettings)
    config = Config(
        **(config_data or {}),
        config_file=str(config_file) if config_file else None,
        profile=profile,
    )

    return config


def merge_configs(base: Config, override: Dict[str, Any]) -> Config:
    """Merge override dict into base config, returning new Config instance."""
    base_dict = base.to_dict()
    _deep_merge(base_dict, override)
    return Config(**base_dict, config_file=base.config_file, profile=base.profile)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
