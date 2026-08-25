"""
Configuration management for MoE Ultra Engine.

Handles loading, validation, and merging of configuration from YAML files,
environment variables, and programmatic overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from copy import deepcopy


@dataclass
class ExpertConfig:
    """Configuration for a single expert in the MoE model."""
    expert_id: int
    layer_id: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    activation_function: str = "silu"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    quantization: str = "int4"  # int4, int8, fp16, fp32
    offload_to_cpu: bool = True
    priority: int = 0  # Higher = keep in memory longer

    def __post_init__(self):
        if self.quantization not in ("int4", "int8", "fp16", "fp32"):
            raise ValueError(f"Invalid quantization: {self.quantization}")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("Attention heads must be positive")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")

    def memory_footprint_mb(self) -> float:
        """Estimate memory footprint in MB for this expert."""
        # Parameters: gate_proj + up_proj + down_proj + attention
        param_count = (
            2 * self.hidden_size * self.intermediate_size +  # gate + up
            self.hidden_size * self.intermediate_size +         # down
            4 * self.hidden_size * self.hidden_size             # qkv + o_proj (approx)
        )
        bytes_per_param = {"int4": 0.5, "int8": 1.0, "fp16": 2.0, "fp32": 4.0}[self.quantization]
        return (param_count * bytes_per_param) / (1024 * 1024)


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    total_ram_gb: float = 32.0
    reserved_ram_gb: float = 4.0  # OS + other processes
    expert_cache_gb: float = 20.0  # For expert weights
    activation_cache_gb: float = 4.0  # For intermediate activations
    kv_cache_gb: float = 4.0  # For KV cache
    swap_dir: str = "/tmp/moe_swap"
    enable_swap: bool = True
    swap_compression: str = "lz4"  # lz4, zstd, none
    prefetch_experts: int = 2  # Number of experts to prefetch
    eviction_policy: str = "lru"  # lru, lfu, priority
    memory_mapped: bool = True

    def __post_init__(self):
        if self.total_ram_gb <= 0:
            raise ValueError("total_ram_gb must be positive")
        if self.reserved_ram_gb >= self.total_ram_gb:
            raise ValueError("reserved_ram_gb must be less than total_ram_gb")
        available = self.total_ram_gb - self.reserved_ram_gb
        allocated = self.expert_cache_gb + self.activation_cache_gb + self.kv_cache_gb
        if allocated > available:
            raise ValueError(f"Allocated memory ({allocated}GB) exceeds available ({available}GB)")
        if self.swap_compression not in ("lz4", "zstd", "none"):
            raise ValueError(f"Invalid swap_compression: {self.swap_compression}")
        if self.eviction_policy not in ("lru", "lfu", "priority"):
            raise ValueError(f"Invalid eviction_policy: {self.eviction_policy}")
        Path(self.swap_dir).mkdir(parents=True, exist_ok=True)

    @property
    def available_ram_gb(self) -> float:
        return self.total_ram_gb - self.reserved_ram_gb


@dataclass
class EngineConfig:
    """Main engine configuration."""
    model_path: str
    model_name: str = "moe-model"
    num_layers: int = 80
    num_experts_per_layer: int = 256
    num_experts_per_token: int = 8
    hidden_size: int = 8192
    intermediate_size: int = 29568
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    vocab_size: int = 151936
    max_sequence_length: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    expert_configs: List[ExpertConfig] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dtype: str = "float16"  # float16, bfloat16, float32
    device: str = "cpu"  # cpu, cuda, mps
    num_threads: int = 0  # 0 = auto
    batch_size: int = 1
    enable_flash_attention: bool = False
    enable_xformers: bool = False
    compile_model: bool = False
    log_level: str = "INFO"
    metrics_enabled: bool = True
    metrics_port: int = 9090

    def __post_init__(self):
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_experts_per_layer <= 0:
            raise ValueError("num_experts_per_layer must be positive")
        if self.num_experts_per_token <= 0 or self.num_experts_per_token > self.num_experts_per_layer:
            raise ValueError("num_experts_per_token must be in [1, num_experts_per_layer]")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"Invalid device: {self.device}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def get_expert_config(self, layer_id: int, expert_id: int) -> ExpertConfig:
        """Get expert config, creating default if not explicitly configured."""
        for ec in self.expert_configs:
            if ec.layer_id == layer_id and ec.expert_id == expert_id:
                return ec
        # Return default config
        return ExpertConfig(
            expert_id=expert_id,
            layer_id=layer_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

    def estimate_total_memory_gb(self) -> float:
        """Estimate total model memory footprint in GB."""
        total_params = 0
        for layer_id in range(self.num_layers):
            for expert_id in range(self.num_experts_per_layer):
                ec = self.get_expert_config(layer_id, expert_id)
                total_params += ec.memory_footprint_mb()
        return total_params / 1024


@dataclass
class MoEConfig:
    """Top-level configuration container."""
    engine: EngineConfig
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> "MoEConfig":
        """Load configuration from YAML file with optional overrides."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)
        
        # Apply programmatic overrides
        if overrides:
            data = cls._deep_merge(data, overrides)
        
        return cls._from_dict(data)
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides (MOE_* prefix)."""
        env_mapping = {
            'MOE_MODEL_PATH': 'engine.model_path',
            'MOE_MODEL_NAME': 'engine.model_name',
            'MOE_NUM_LAYERS': 'engine.num_layers',
            'MOE_NUM_EXPERTS_PER_LAYER': 'engine.num_experts_per_layer',
            'MOE_NUM_EXPERTS_PER_TOKEN': 'engine.num_experts_per_token',
            'MOE_HIDDEN_SIZE': 'engine.hidden_size',
            'MOE_INTERMEDIATE_SIZE': 'engine.intermediate_size',
            'MOE_NUM_ATTENTION_HEADS': 'engine.num_attention_heads',
            'MOE_NUM_KEY_VALUE_HEADS': 'engine.num_key_value_heads',
            'MOE_VOCAB_SIZE': 'engine.vocab_size',
            'MOE_MAX_SEQUENCE_LENGTH': 'engine.max_sequence_length',
            'MOE_ROPE_THETA': 'engine.rope_theta',
            'MOE_DTYPE': 'engine.dtype',
            'MOE_DEVICE': 'engine.device',
            'MOE_NUM_THREADS': 'engine.num_threads',
            'MOE_BATCH_SIZE': 'engine.batch_size',
            'MOE_TOTAL_RAM_GB': 'engine.memory.total_ram_gb',
            'MOE_RESERVED_RAM_GB': 'engine.memory.reserved_ram_gb',
            'MOE_EXPERT_CACHE_GB': 'engine.memory.expert_cache_gb',
            'MOE_ACTIVATION_CACHE_GB': 'engine.memory.activation_cache_gb',
            'MOE_KV_CACHE_GB': 'engine.memory.kv_cache_gb',
            'MOE_SWAP_DIR': 'engine.memory.swap_dir',
            'MOE_ENABLE_SWAP': 'engine.memory.enable_swap',
            'MOE_SWAP_COMPRESSION': 'engine.memory.swap_compression',
            'MOE_PREFETCH_EXPERTS': 'engine.memory.prefetch_experts',
            'MOE_EVICTION_POLICY': 'engine.memory.eviction_policy',
            'MOE_LOG_LEVEL': 'engine.log_level',
            'MOE_METRICS_ENABLED': 'engine.metrics_enabled',
            'MOE_METRICS_PORT': 'engine.metrics_port',
        }
        
        result = deepcopy(data)
        for env_var, config_path in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                cls._set_nested(result, config_path.split('.'), cls._parse_value(value))
        return result
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        # JSON
        if value.startswith('{') or value.startswith('['):
            import json
            return json.loads(value)
        return value
    
    @staticmethod
    def _set_nested(data: Dict[str, Any], keys: List[str], value: Any) -> None:
        """Set nested dictionary value."""
        for key in keys[:-1]:
            data = data.setdefault(key, {})
        data[keys[-1]] = value
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = MoEConfig._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "MoEConfig":
        """Create MoEConfig from dictionary."""
        engine_data = data.get('engine', {})
        
        # Parse expert configs
        expert_configs = []
        for ec_data in engine_data.get('expert_configs', []):
            expert_configs.append(ExpertConfig(**ec_data))
        
        # Parse memory config
        memory_data = engine_data.get('memory', {})
        memory = MemoryConfig(**memory_data)
        
        # Create engine config
        engine = EngineConfig(
            expert_configs=expert_configs,
            memory=memory,
            **{k: v for k, v in engine_data.items() if k not in ('expert_configs', 'memory')}
        )
        
        return cls(engine=engine)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'engine': {
                **asdict(self.engine),
                'expert_configs': [asdict(ec) for ec in self.engine.expert_configs],
                'memory': asdict(self.engine.memory),
            }
        }
