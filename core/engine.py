"""
MoE Ultra Engine - Core Inference Engine

Ultra-memory-efficient inference engine for Mixture-of-Experts (MoE) models.
Capable of running 2.4T parameter models on 32GB RAM.
"""

import os
import sys
import mmap
import struct
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    safe_open = None

from .config import EngineConfig, ModelConfig, ExpertConfig
from .memory_manager import MemoryManager
from .expert_cache import ExpertCache, CacheEntry
from .router import Router, RoutingResult
from .quantization import Quantizer, QuantizationConfig
from .moe_model import MoEModel, MoELayer

logger = logging.getLogger(__name__)


class InferenceState(Enum):
    """Engine inference states."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    INFERRING = "inferring"
    ERROR = "error"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_strings: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result of an inference request."""
    tokens: List[int]
    text: str
    routing_stats: Dict[str, Any]
    latency_ms: float
    tokens_per_second: float
    memory_used_mb: float
    expert_activations: Dict[int, int]


class MoEUltraEngine:
    """
    Ultra-memory-efficient MoE Inference Engine.
    
    Features:
    - Memory-mapped model loading (no full model in RAM)
    - Expert-level caching with LRU eviction
    - Dynamic expert routing with top-k selection
    - Quantization support (INT4, INT8, FP8)
    - Offloading to CPU/disk for large models
    - Streaming generation support
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state = InferenceState.IDLE
        self.model: Optional[MoEModel] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.expert_cache: Optional[ExpertCache] = None
        self.router: Optional[Router] = None
        self.quantizer: Optional[Quantizer] = None
        self.tokenizer = None
        self._lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._stats = {
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'expert_activations': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'offload_count': 0
        }
        
    def initialize(self) -> bool:
        """Initialize the engine components."""
        with self._lock:
            if self.state != InferenceState.IDLE:
                logger.warning(f"Engine already initialized, state: {self.state}")
                return False
            
            self.state = InferenceState.LOADING
            logger.info("Initializing MoE Ultra Engine...")
            
            try:
                # Initialize memory manager
                self.memory_manager = MemoryManager(
                    max_memory_mb=self.config.max_memory_mb,
                    offload_dir=self.config.offload_dir,
                    enable_mmap=self.config.enable_mmap
                )
                
                # Initialize expert cache
                self.expert_cache = ExpertCache(
                    max_size_mb=self.config.cache_size_mb,
                    memory_manager=self.memory_manager
                )
                
                # Initialize router
                self.router = Router(
                    num_experts=self.config.model.num_experts,
                    top_k=self.config.model.top_k,
                    routing_strategy=self.config.model.routing_strategy
                )
                
                # Initialize quantizer
                if self.config.quantization.enabled:
                    self.quantizer = Quantizer(QuantizationConfig(
                        dtype=self.config.quantization.dtype,
                        group_size=self.config.quantization.group_size,
                        symmetric=self.config.quantization.symmetric
                    ))
                
                self.state = InferenceState.READY
                logger.info("MoE Ultra Engine initialized successfully")
                return True
                
            except Exception as e:
                self.state = InferenceState.ERROR
                logger.error(f"Failed to initialize engine: {e}")
                raise
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load MoE model from path."""
        with self._lock:
            if self.state not in (InferenceState.READY, InferenceState.ERROR):
                raise RuntimeError(f"Engine not ready for model loading, state: {self.state}")
            
            self.state = InferenceState.LOADING
            logger.info(f"Loading model from {model_path}")
            
            try:
                model_path = Path(model_path)
                
                # Load model configuration
                model_config = self._load_model_config(model_path)
                
                # Create MoE model with memory-mapped weights
                self.model = MoEModel(
                    config=model_config,
                    memory_manager=self.memory_manager,
                    expert_cache=self.expert_cache,
                    router=self.router,
                    quantizer=self.quantizer
                )
                
                # Load weights using memory mapping
                self.model.load_weights(model_path)
                
                # Load tokenizer
                self.tokenizer = self._load_tokenizer(model_path)
                
                self.state = InferenceState.READY
                logger.info("Model loaded successfully")
                return True
                
            except Exception as e:
                self.state = InferenceState.ERROR
                logger.error(f"Failed to load model: {e}")
                raise
    
    def _load_model_config(self, model_path: Path) -> ModelConfig:
        """Load model configuration from config.json."""
        import json
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        with open(config_path) as f:
            config_data = json.load(f)
        
        return ModelConfig(
            model_type=config_data.get("model_type", "moe"),
            hidden_size=config_data.get("hidden_size", 4096),
            num_layers=config_data.get("num_hidden_layers", 32),
            num_experts=config_data.get("num_experts", 8),
            num_experts_per_tok=config_data.get("num_experts_per_tok", 2),
            intermediate_size=config_data.get("intermediate_size", 14336),
            vocab_size=config_data.get("vocab_size", 151936),
            max_position_embeddings=config_data.get("max_position_embeddings", 32768),
            rms_norm_eps=config_data.get("rms_norm_eps", 1e-6),
            rope_theta=config_data.get("rope_theta", 1000000.0),
            routing_strategy=config_data.get("routing_strategy", "topk"),
            top_k=config_data.get("num_experts_per_tok", 2),
            expert_capacity_factor=config_data.get("expert_capacity_factor", 1.25),
            tie_word_embeddings=config_data.get("tie_word_embeddings", False)
        )
    
    def _load_tokenizer(self, model_path: Path):
        """Load tokenizer from model directory."""
        try:
            from tokenizers import Tokenizer
            tokenizer_path = model_path / "tokenizer.json"
            if tokenizer_path.exists():
                return Tokenizer.from_file(str(tokenizer_path))
        except ImportError:
            logger.warning("tokenizers library not available")
        
        # Fallback to basic tokenizer
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(model_path))
        except ImportError:
            logger.warning("transformers library not available")
        
        return None
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> InferenceResult:
        """Generate text from prompt."""
        with self._generation_lock:
            if self.state != InferenceState.READY:
                raise RuntimeError(f"Engine not ready, state: {self.state}")
            if self.model is None:
                raise RuntimeError("No model loaded")
            if self.tokenizer is None:
                raise RuntimeError("No tokenizer loaded")
            
            self.state = InferenceState.INFERRING
            start_time = time.perf_counter()
            
            try:
                gen_config = config or GenerationConfig()
                
                # Encode prompt
                input_ids = self._encode(prompt)
                
                # Generate tokens
                generated_tokens, routing_stats = self._generate_tokens(
                    input_ids, gen_config
                )
                
                # Decode generated text
                generated_text = self._decode(generated_tokens)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                tokens_per_second = len(generated_tokens) / (latency_ms / 1000) if latency_ms > 0 else 0
                
                # Update stats
                self._stats['total_tokens'] += len(generated_tokens)
                self._stats['total_latency_ms'] += latency_ms
                
                result = InferenceResult(
                    tokens=generated_tokens,
                    text=generated_text,
                    routing_stats=routing_stats,
                    latency_ms=latency_ms,
                    tokens_per_second=tokens_per_second,
                    memory_used_mb=self.memory_manager.get_used_memory_mb() if self.memory_manager else 0,
                    expert_activations=self._stats['expert_activations'].copy()
                )
                
                self.state = InferenceState.READY
                return result
                
            except Exception as e:
                self.state = InferenceState.ERROR
                logger.error(f"Generation failed: {e}")
                raise
    
    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text).ids
        elif hasattr(self.tokenizer, '__call__'):
            return self.tokenizer(text).input_ids
        else:
            # Fallback: simple character-level encoding
            return [ord(c) for c in text]
    
    def _decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids)
        else:
            # Fallback: simple character-level decoding
            return ''.join(chr(max(0, min(t, 0x10FFFF))) for t in token_ids)
    
    def _generate_tokens(self, input_ids: List[int], config: GenerationConfig) -> Tuple[List[int], Dict[str, Any]]:
        """Core token generation loop."""
        generated = []
        routing_stats = {
            'layer_routing': [],
            'expert_counts': {},
            'total_routed': 0
        }
        
        # Prepare input tensor
        if TORCH_AVAILABLE:
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
        else:
            input_tensor = np.array([input_ids], dtype=np.int64)
        
        past_key_values = None
        
        for step in range(config.max_new_tokens):
            # Forward pass through model
            logits, past_key_values, step_routing = self.model.forward(
                input_tensor if step == 0 else input_tensor[:, -1:],
                past_key_values=past_key_values
            )
            
            # Update routing stats
            routing_stats['layer_routing'].append(step_routing)
            for layer_idx, experts in step_routing.items():
                for expert_idx in experts:
                    routing_stats['expert_counts'][expert_idx] = routing_stats['expert_counts'].get(expert_idx, 0) + 1
                    self._stats['expert_activations'][expert_idx] = self._stats['expert_activations'].get(expert_idx, 0) + 1
            routing_stats['total_routed'] += sum(len(e) for e in step_routing.values())
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0 and generated:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )
            
            # Sample next token
            next_token = self._sample_token(next_token_logits, config)
            
            generated.append(next_token)
            
            # Check for EOS
            if config.eos_token_id and next_token == config.eos_token_id:
                break
            
            # Check for stop strings
            if config.stop_strings:
                current_text = self._decode(generated)
                if any(stop in current_text for stop in config.stop_strings):
                    break
            
            # Update input tensor for next step
            if TORCH_AVAILABLE:
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            else:
                input_tensor = np.concatenate([input_tensor, np.array([[next_token]], dtype=np.int64)], axis=1)
        
        return generated, routing_stats
    
    def _apply_repetition_penalty(self, logits, generated_tokens: List[int], penalty: float):
        """Apply repetition penalty to logits."""
        if TORCH_AVAILABLE and isinstance(logits, torch.Tensor):
            for token in set(generated_tokens):
                logits[:, token] /= penalty
        else:
            for token in set(generated_tokens):
                logits[:, token] /= penalty
        return logits
    
    def _sample_token(self, logits, config: GenerationConfig) -> int:
        """Sample next token from logits."""
        if TORCH_AVAILABLE and isinstance(logits, torch.Tensor):
            probs = torch.softmax(logits, dim=-1)
            
            if config.do_sample:
                # Top-k filtering
                if config.top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, config.top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                
                # Top-p (nucleus) filtering
                if config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(probs, dim=-1).item()
        else:
            # NumPy fallback
            probs = self._softmax(logits[0])
            
            if config.do_sample:
                # Top-k
                if config.top_k > 0:
                    top_k_indices = np.argpartition(probs, -config.top_k)[-config.top_k:]
                    mask = np.zeros_like(probs, dtype=bool)
                    mask[top_k_indices] = True
                    probs = probs * mask
                
                # Top-p
                if config.top_p < 1.0:
                    sorted_indices = np.argsort(probs)[::-1]
                    sorted_probs = probs[sorted_indices]
                    cumulative_probs = np.cumsum(sorted_probs)
                    cutoff = np.searchsorted(cumulative_probs, config.top_p)
                    probs[sorted_indices[cutoff:]] = 0
                
                probs = probs / probs.sum()
                next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = int(np.argmax(probs))
        
        return next_token
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def stream_generate(self, prompt: str, config: Optional[GenerationConfig] = None):
        """Stream tokens as they are generated."""
        if self.state != InferenceState.READY:
            raise RuntimeError(f"Engine not ready, state: {self.state}")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")
        
        gen_config = config or GenerationConfig()
        input_ids = self._encode(prompt)
        
        if TORCH_AVAILABLE:
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
        else:
            input_tensor = np.array([input_ids], dtype=np.int64)
        
        past_key_values = None
        generated = []
        
        for step in range(gen_config.max_new_tokens):
            logits, past_key_values, _ = self.model.forward(
                input_tensor if step == 0 else input_tensor[:, -1:],
                past_key_values=past_key_values
            )
            
            next_token_logits = logits[:, -1, :]
            
            if gen_config.temperature != 1.0:
                next_token_logits = next_token_logits / gen_config.temperature
            
            if gen_config.repetition_penalty != 1.0 and generated:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, gen_config.repetition_penalty
                )
            
            next_token = self._sample_token(next_token_logits, gen_config)
            generated.append(next_token)
            
            yield self._decode([next_token])
            
            if gen_config.eos_token_id and next_token == gen_config.eos_token_id:
                break
            
            if TORCH_AVAILABLE:
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            else:
                input_tensor = np.concatenate([input_tensor, np.array([[next_token]], dtype=np.int64)], axis=1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        cache_stats = {}
        if self.expert_cache:
            cache_stats = self.expert_cache.get_stats()
        
        memory_stats = {}
        if self.memory_manager:
            memory_stats = self.memory_manager.get_stats()
        
        return {
            'engine_state': self.state.value,
            'total_tokens_generated': self._stats['total_tokens'],
            'average_latency_ms': self._stats['total_latency_ms'] / max(1, self._stats['total_tokens']),
            'expert_activations': self._stats['expert_activations'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'offload_count': self._stats['offload_count'],
            'cache': cache_stats,
            'memory': memory_stats
        }
    
    def shutdown(self):
        """Shutdown engine and release resources."""
        with self._lock:
            logger.info("Shutting down MoE Ultra Engine...")
            
            if self.expert_cache:
                self.expert_cache.clear()
            
            if self.memory_manager:
                self.memory_manager.cleanup()
            
            self.model = None
            self.tokenizer = None
            self.state = InferenceState.IDLE
            logger.info("Engine shutdown complete")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


def create_engine(config_path: Optional[str] = None) -> MoEUltraEngine:
    """Factory function to create engine from config file."""
    from .config import load_config
    config = load_config(config_path) if config_path else EngineConfig()
    return MoEUltraEngine(config)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    config = EngineConfig()
    engine = MoEUltraEngine(config)
    print("MoE Ultra Engine created successfully")
    print(f"Config: {config}")