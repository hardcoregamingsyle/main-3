"""
Configuration management for MoE Ultra Engine.
Supports YAML config files, environment variables, and programmatic configuration.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import yaml


class QuantizationType(str, Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP8 = "fp8"
    GPTQ = "gptq"
    AWQ = "awq"


class DeviceType(str, Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class QuantizationConfig:
    """Quantization configuration for model weights."""
    type: QuantizationType = QuantizationType.INT4
    group_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = False
    static_groups: bool = False
    sym: bool = True
    true_sequential: bool = True
    bits: int = 4

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = QuantizationType(self.type.lower())
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported bits: {self.bits}. Must be 2, 3, 4, or 8")
        if self.group_size not in (32, 64, 128, 256, -1):
            raise ValueError(f"Unsupported group_size: {self.group_size}")


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str
    path: str
    model_type: str = "moe"
    num_experts: int = 256
    num_experts_per_token: int = 8
    hidden_size: int = 8192
    intermediate_size: int = 28672
    num_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    torch_dtype: str = "float16"
    quantization: Optional[QuantizationConfig] = None
    offload_folder: Optional[str] = None
    use_flash_attn: bool = True
    use_sdpa: bool = True
    expert_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = True
    revision: str = "main"

    def __post_init__(self):
        if isinstance(self.quantization, dict):
            self.quantization = QuantizationConfig(**self.quantization)
        if self.quantization is None:
            self.quantization = QuantizationConfig()


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_memory_gb: float = 32.0
    cpu_offload_gb: float = 24.0
    gpu_memory_gb: float = 0.0
    kv_cache_gb: float = 4.0
    activation_gb: float = 2.0
    expert_cache_gb: float = 8.0
    buffer_gb: float = 1.0
    enable_memory_mapping: bool = True
    enable_cpu_offload: bool = True
    enable_pinned_memory: bool = True
    prefetch_experts: int = 4
    expert_cache_policy: str = "lru"  # lru, lfu, fifo

    def __post_init__(self):
        total = self.cpu_offload_gb + self.gpu_memory_gb + self.kv_cache_gb + \
                self.activation_gb + self.expert_cache_gb + self.buffer_gb
        if total > self.max_memory_gb * 1.1:  # 10% tolerance
            raise ValueError(f"Memory allocation ({total:.1f}GB) exceeds max ({self.max_memory_gb}GB)")


@dataclass
class InferenceConfig:
    """Inference runtime configuration."""
    max_batch_size: int = 1
    max_sequence_length: int = 32768
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    use_cache: bool = True
    return_dict_in_generate: bool = True
    output_scores: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
    stream: bool = False
    seed: Optional[int] = None


@dataclass
class EngineConfig:
    """Main engine configuration."""
    model: ModelConfig
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: DeviceType = DeviceType.AUTO
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_profiling: bool = False
    profile_output_dir: str = "./profiles"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    enable_xformers: bool = False
    enable_triton: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Create configuration from dictionary."""
        model_data = data.get('model', {})
        if isinstance(model_data, dict):
            model_config = ModelConfig(**model_data)
        else:
            model_config = model_data
        
        memory_config = MemoryConfig(**data.get('memory', {}))
        inference_config = InferenceConfig(**data.get('inference', {}))
        
        device = data.get('device', DeviceType.AUTO)
        if isinstance(device, str):
            device = DeviceType(device.lower())
        
        return cls(
            model=model_config,
            memory=memory_config,
            inference=inference_config,
            device=device,
            log_level=data.get('log_level', 'INFO'),
            log_file=data.get('log_file'),
            enable_profiling=data.get('enable_profiling', False),
            profile_output_dir=data.get('profile_output_dir', './profiles'),
            compile_model=data.get('compile_model', False),
            compile_mode=data.get('compile_mode', 'reduce-overhead'),
            enable_xformers=data.get('enable_xformers', False),
            enable_triton=data.get('enable_triton', True),
            num_workers=data.get('num_workers', 4),
            prefetch_factor=data.get('prefetch_factor', 2),
            persistent_workers=data.get('persistent_workers', True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'memory': self.memory.__dict__,
            'inference': self.inference.__dict__,
            'device': self.device.value,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'enable_profiling': self.enable_profiling,
            'profile_output_dir': self.profile_output_dir,
            'compile_model': self.compile_model,
            'compile_mode': self.compile_mode,
            'enable_xformers': self.enable_xformers,
            'enable_triton': self.enable_triton,
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor,
            'persistent_workers': self.persistent_workers,
        }

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[str] = None) -> EngineConfig:
    """Load configuration from file or environment."""
    if config_path and os.path.exists(config_path):
        return EngineConfig.from_yaml(config_path)
    
    # Check environment variable
    env_config = os.environ.get('MOE_ENGINE_CONFIG')
    if env_config and os.path.exists(env_config):
        return EngineConfig.from_yaml(env_config)
    
    # Check default locations
    default_paths = [
        'config/default.yaml',
        'config/prod.yaml',
        './config.yaml',
        './config.yml',
    ]
    for path in default_paths:
        if os.path.exists(path):
            return EngineConfig.from_yaml(path)
    
    # Return default configuration
    return EngineConfig(
        model=ModelConfig(
            name="default",
            path="./models",
        )
    )
