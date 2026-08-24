"""Configuration management for MoE Ultra Engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_config_path(env: str = "development") -> str:
    """Get the configuration file path for the given environment.
    
    Args:
        env: Environment name
        
    Returns:
        Path to configuration file
    """
    base_dir = Path(__file__).parent.parent
    config_file = base_dir / "config" / f"{env}.yaml"
    return str(config_file)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Merge with environment variables (overrides)
    env_prefix = "MOE_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            nested_key = key[len(env_prefix):].lower()
            parts = nested_key.split("_")
            
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
    
    return config


class ServerConfig(BaseModel):
    """Server configuration settings."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, ge=1, le=65535, description="Port to bind to")
    workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    log_level: str = Field(default="info", description="Logging level")


class InferenceConfig(BaseModel):
    """Inference engine configuration."""
    model_path: str = Field(..., description="Path to model files")
    device: str = Field(default="cpu", description="Device to run on (cpu, cuda, mps)")
    precision: str = Field(default="bf16", description="Precision mode (fp32, fp16, bf16, int8, int4)")
    max_context_length: int = Field(default=4096, ge=128, le=32768, description="Maximum context length")
    max_batch_size: int = Field(default=8, ge=1, le=128, description="Maximum batch size")
    num_experts: int = Field(default=8, ge=1, le=64, description="Number of experts to load")
    active_experts: int = Field(default=2, ge=1, le=8, description="Active experts per token")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=40, ge=1, le=500, description="Top-k sampling")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    
    class Config:
        protected_namespaces = ()


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_pre_ping: bool = Field(default=True, description="Enable connection health checks")


class RedisConfig(BaseModel):
    """Redis configuration settings."""
    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max connection pool size")
    socket_timeout: float = Field(default=5.0, ge=0.1, description="Socket timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")


class CacheConfig(BaseModel):
    """Cache configuration settings."""
    enabled: bool = Field(default=True, description="Enable caching")
    ttl: int = Field(default=3600, ge=60, le=86400, description="Default TTL in seconds")
    max_size: int = Field(default=10000, ge=100, le=1000000, description="Max cache entries")
    prefix: str = Field(default="moe:", description="Cache key prefix")


class MonitoringConfig(BaseModel):
    """Monitoring configuration settings."""
    enabled: bool = Field(default=True, description="Enable metrics collection")
    endpoint: str = Field(default="/metrics", description="Prometheus metrics endpoint")
    bucket_ranges: list[float] = Field(
        default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        description="Histogram bucket ranges"
    )


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="structured", description="Log format (structured, console)")
    file_enabled: bool = Field(default=False, description="Enable file logging")
    file_path: str = Field(default="logs/app.log", description="Log file path")
    file_level: str = Field(default="INFO", description="File log level")
    structured: bool = Field(default=False, description="Use JSON structured logging")
    colors: bool = Field(default=True, description="Use colored output in console")


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    jwt_secret: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    rate_limit_requests: int = Field(default=100, ge=1, le=10000, description="Rate limit requests")
    rate_limit_window: int = Field(default=60, ge=1, le=3600, description="Rate limit window seconds")


class Config(BaseSettings):
    """Main configuration model."""
    model_config = SettingsConfigDict(
        env_prefix="MOE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    database: Optional[DatabaseConfig] = None
    redis: Optional[RedisConfig] = None
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    @validator("debug", pre=True)
    def parse_debug(cls, v: Any) -> bool:
        """Parse debug flag from string."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Create Config instance from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        config_dict = load_config(config_path)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation
        """
        return self.model_dump(mode="json")


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Global Config instance
    """
    global _config
    
    if _config is None:
        if config_path:
            _config = Config.from_yaml(config_path)
        else:
            env = os.getenv("MOE_ENVIRONMENT", "development")
            config_path = get_config_path(env)
            _config = Config.from_yaml(config_path)
    
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload the global configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Reloading Config instance
    """
    global _config
    _config = None
    return get_config(config_path)
