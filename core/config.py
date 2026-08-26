"""
Configuration management for MoE Ultra Engine.

Supports YAML config files, environment variable overrides, and validation.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from functools import lru_cache


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str = "qwen-3.8-max"
    path: str = "./models/qwen-3.8-max"
    dtype: str = "fp16"  # fp16, bf16, int8, int4
    context_length: int = 32768
    vocab_size: int = 151936
    hidden_size: int = 8192
    num_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    intermediate_size: int = 29568
    num_experts: int = 256
    experts_per_token: int = 8
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "linear", "factor": 1.0}
        valid_dtypes = ["fp32", "fp16", "bf16", "int8", "int4"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype}")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token cannot exceed num_experts")


@dataclass
class EngineConfig:
    """Inference engine configuration."""
    max_batch_size: int = 1
    max_sequence_length: int = 32768
    kv_cache_dtype: str = "fp16"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    chunked_prefill_size: int = 2048
    enable_speculative_decoding: bool = False
    speculative_draft_model: Optional[str] = None
    num_speculative_tokens: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 2048
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


@dataclass
class HardwareConfig:
    """Hardware and memory configuration."""
    device: str = "cpu"  # cpu, cuda, mps, xpu
    cpu_threads: int = 0  # 0 = auto
    memory_limit_gb: float = 32.0
    offload_folder: str = "./offload"
    enable_mmap: bool = True
    enable_mlock: bool = False
    numa_nodes: List[int] = field(default_factory=list)
    gpu_layers: int = 0
    gpu_memory_fraction: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    
    def __post_init__(self):
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be > 0")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        valid_devices = ["cpu", "cuda", "mps", "xpu"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    metrics_enabled: bool = True
    metrics_port: int = 9090
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        model_data = data.get('model', {})
        engine_data = data.get('engine', {})
        hardware_data = data.get('hardware', {})
        
        return cls(
            model=ModelConfig(**model_data),
            engine=EngineConfig(**engine_data),
            hardware=HardwareConfig(**hardware_data),
            log_level=data.get('log_level', 'INFO'),
            log_format=data.get('log_format', 'json'),
            log_file=data.get('log_file'),
            metrics_enabled=data.get('metrics_enabled', True),
            metrics_port=data.get('metrics_port', 9090),
            api_host=data.get('api_host', '0.0.0.0'),
            api_port=data.get('api_port', 8000),
            api_workers=data.get('api_workers', 1),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def apply_env_overrides(self) -> "Config":
        """Apply environment variable overrides."""
        env_mappings = {
            'MOE_MODEL_PATH': ('model', 'path'),
            'MOE_MODEL_NAME': ('model', 'name'),
            'MOE_DTYPE': ('model', 'dtype'),
            'MOE_CONTEXT_LENGTH': ('model', 'context_length', int),
            'MOE_NUM_EXPERTS': ('model', 'num_experts', int),
            'MOE_EXPERTS_PER_TOKEN': ('model', 'experts_per_token', int),
            'MOE_MAX_BATCH_SIZE': ('engine', 'max_batch_size', int),
            'MOE_TEMPERATURE': ('engine', 'temperature', float),
            'MOE_TOP_P': ('engine', 'top_p', float),
            'MOE_MAX_NEW_TOKENS': ('engine', 'max_new_tokens', int),
            'MOE_DEVICE': ('hardware', 'device'),
            'MOE_CPU_THREADS': ('hardware', 'cpu_threads', int),
            'MOE_MEMORY_LIMIT_GB': ('hardware', 'memory_limit_gb', float),
            'MOE_OFFLOAD_FOLDER': ('hardware', 'offload_folder'),
            'MOE_GPU_LAYERS': ('hardware', 'gpu_layers', int),
            'MOE_LOG_LEVEL': (None, 'log_level'),
            'MOE_API_HOST': (None, 'api_host'),
            'MOE_API_PORT': (None, 'api_port', int),
            'MOE_API_WORKERS': (None, 'api_workers', int),
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if len(mapping) == 3:
                    section, key, converter = mapping
                    value = converter(value)
                else:
                    section, key = mapping
                
                if section is None:
                    setattr(self, key, value)
                else:
                    section_obj = getattr(self, section)
                    setattr(section_obj, key, value)
        
        return self


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance."""
    if config_path is None:
        config_path = os.environ.get('MOE_CONFIG_PATH', 'config/default.yaml')
    
    config_path = Path(config_path)
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()
    
    return config.apply_env_overrides()


def validate_config(config: Config) -> List[str]:
    """Validate configuration and return list of warnings."""
    warnings = []
    
    # Check model path exists
    model_path = Path(config.model.path)
    if not model_path.exists():
        warnings.append(f"Model path does not exist: {model_path}")
    
    # Check memory requirements
    estimated_memory = estimate_memory_requirements(config)
    if estimated_memory > config.hardware.memory_limit_gb:
        warnings.append(
            f"Estimated memory ({estimated_memory:.1f}GB) exceeds limit "
            f"({config.hardware.memory_limit_gb}GB)"
        )
    
    # Check offload folder
    offload_path = Path(config.hardware.offload_folder)
    if not offload_path.exists():
        try:
            offload_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(f"Cannot create offload folder: {e}")
    
    return warnings


def estimate_memory_requirements(config: Config) -> float:
    """Estimate memory requirements in GB."""
    model = config.model
    hardware = config.hardware
    
    # Parameter count estimation
    # Embedding: vocab_size * hidden_size
    embedding_params = model.vocab_size * model.hidden_size
    
    # Per layer: attention + FFN (MoE)
    # Attention: 4 * hidden_size^2 (Q, K, V, O projections)
    attention_params = 4 * model.hidden_size * model.hidden_size
    
    # MoE FFN: num_experts * 2 * hidden_size * intermediate_size (gate + up + down)
    # Actually: gate_proj + up_proj + down_proj per expert
    ffn_params_per_expert = 3 * model.hidden_size * model.intermediate_size
    moe_params = model.num_experts * ffn_params_per_expert
    
    # Layer norm params (2 per layer)
    ln_params = 2 * model.hidden_size
    
    total_params_per_layer = attention_params + moe_params + ln_params
    total_params = embedding_params + model.num_layers * total_params_per_layer
    
    # Memory per parameter based on dtype
    dtype_bytes = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5,
    }
    bytes_per_param = dtype_bytes.get(model.dtype, 2)
    
    model_memory_gb = (total_params * bytes_per_param) / (1024**3)
    
    # KV cache memory
    # 2 * num_layers * num_kv_heads * head_dim * max_seq_len * batch_size * bytes
    head_dim = model.hidden_size // model.num_attention_heads
    kv_cache_gb = (
        2 * model.num_layers * model.num_key_value_heads * head_dim *
        config.engine.max_sequence_length * config.engine.max_batch_size * bytes_per_param
    ) / (1024**3)
    
    # Activation memory (rough estimate)
    activation_gb = (
        config.engine.max_batch_size * config.engine.max_sequence_length *
        model.hidden_size * bytes_per_param * 4  # factor for intermediate activations
    ) / (1024**3)
    
    total = model_memory_gb + kv_cache_gb + activation_gb
    
    # Add overhead for offloading structures
    total *= 1.15
    
    return total
