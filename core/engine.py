"""
Main MoE Inference Engine for ultra-efficient model execution.

Handles expert routing, memory management, and token generation for
Mixture-of-Experts models optimized for consumer hardware.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from core.config import Config
from core.model_loader import ModelLoader
from core.quantization import Quantizer


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Inference execution modes."""
    BATCHED = "batched"
    STREAMING = "streaming"
    ASYNC = "async"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)
    do_sample: bool = True


@dataclass
class ExpertStats:
    """Statistics for expert usage."""
    expert_id: int
    calls: int = 0
    latency_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class GenerationResult:
    """Result of a generation request."""
    text: str
    tokens: List[int]
    logprobs: List[float]
    timing: Dict[str, float]
    stats: Dict[str, Any]


class MoeEngine:
    """
    Ultra-memory-efficient MoE inference engine.
    
    Optimized for running large parameter-count models (up to 2.4T)
    on consumer hardware with limited RAM (32GB DDR4/DDR3).
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.quantizer = Quantizer(config)
        
        # Core state
        self._model_loaded = False
        self._active_session = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.expert_stats: Dict[int, ExpertStats] = {}
        self.token_times: List[float] = []
        self.total_tokens_generated = 0
        
        # Memory management
        self._memory_pool: Dict[str, np.ndarray] = {}
        self.max_memory_mb = config.get("max_memory_mb", 30 * 1024)
        
        logger.info(f"MoeEngine initialized with config: {config}")
    
    async def load_model(self, model_path: str, quantization: str = "int4") -> bool:
        """Load a MoE model from disk."""
        if self._model_loaded:
            logger.warning("Model already loaded, skipping")
            return True
            
        logger.info(f"Loading MoE model from {model_path} with quantization: {quantization}")
        
        try:
            # Load model weights
            await self.model_loader.load(model_path, quantization)
            
            # Initialize expert statistics
            num_experts = len(self.model_loader.expert_map)
            self.expert_stats = {i: ExpertStats(expert_id=i) for i in range(num_experts)}
            
            self._model_loaded = True
            logger.info(f"Model loaded successfully. Experts: {num_experts}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        mode: ExecutionMode = ExecutionMode.STREAMING
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from a prompt."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        # Tokenize input
        tokens = self.model_loader.tokenize(prompt)
        
        # Generate tokens
        generated_tokens = []
        generated_text = ""
        
        if mode == ExecutionMode.BATCHED:
            result = await self._generate_batch(tokens, config)
            generated_tokens = result.tokens
            generated_text = result.text
            
        elif mode == ExecutionMode.STREAMING:
            generator = self._generate_stream(tokens, config)
            async for chunk in generator:
                yield chunk
                
        elif mode == ExecutionMode.ASYNC:
            result = await self._generate_async(tokens, config)
            generated_tokens = result.tokens
            generated_text = result.text
        
        # Update statistics
        end_time = time.time()
        total_time = end_time - start_time
        avg_token_time = total_time / len(generated_tokens) if generated_tokens else 0
        
        result_dict = {
            "text": generated_text,
            "tokens": generated_tokens,
            "logprobs": [],  # Would be populated with actual logprobs
            "timing": {
                "total_seconds": round(total_time, 3),
                "tokens_per_second": round(1 / avg_token_time, 2) if avg_token_time > 0 else 0,
                "first_token_seconds": 0.0  # Would track TTFB
            },
            "stats": {
                "experts_used": list(self.expert_stats.keys()),
                "total_expert_calls": sum(s.calls for s in self.expert_stats.values())
            }
        }
        
        return result_dict
    
    async def _generate_batch(
        self,
        tokens: List[int],
        config: GenerationConfig
    ) -> GenerationResult:
        """Generate all tokens in batch mode."""
        generated_tokens = tokens.copy()
        max_new_tokens = config.max_tokens
        
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            # Get current context
            context = generated_tokens[-min(len(generated_tokens), 512):]
            
            # Run forward pass through experts
            expert_outputs = self._route_experts(context)
            
            # Sample next token
            next_token = self._sample_token(expert_outputs, config)
            
            # Handle stop conditions
            if next_token in self.model_loader.stop_ids:
                break
                
            generated_tokens.append(next_token)
            
            # Track expert usage
            for expert_id, output in expert_outputs.items():
                self.expert_stats[expert_id].calls += 1
            
            # Memory pressure check
            if len(generated_tokens) % 100 == 0:
                self._check_memory_pressure()
        
        # Decode tokens to text
        text = self.model_loader.decode(generated_tokens[len(tokens):])
        
        return GenerationResult(
            text=text,
            tokens=generated_tokens,
            logprobs=[],
            timing={},
            stats={}
        )
    
    async def _generate_stream(
        self,
        tokens: List[int],
        config: GenerationConfig
    ) -> AsyncGenerator[str, None]:
        """Stream generation token by token."""
        generated_tokens = tokens.copy()
        max_new_tokens = config.max_tokens
        
        for step in range(max_new_tokens):
            # Get current context
            context = generated_tokens[-min(len(generated_tokens), 512):]
            
            # Run forward pass through experts
            expert_outputs = self._route_experts(context)
            
            # Sample next token
            next_token = self._sample_token(expert_outputs, config)
            
            # Handle stop conditions
            if next_token in self.model_loader.stop_ids:
                break
            
            # Convert token to text chunk
            token_text = self.model_loader.decode([next_token])
            generated_tokens.append(next_token)
            
            # Yield chunk
            yield token_text
            
            # Track expert usage
            for expert_id, output in expert_outputs.items():
                self.expert_stats[expert_id].calls += 1
            
            # Small delay for streaming effect
            await asyncio.sleep(0.001)
    
    async def _generate_async(
        self,
        tokens: List[int],
        config: GenerationConfig
    ) -> GenerationResult:
        """Generate tokens asynchronously."""
        # Create async tasks for parallel expert computation
        tasks = [
            self._compute_expert_async(i, tokens)
            for i in range(len(self.model_loader.expert_map))
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        aggregated = self._aggregate_expert_results(results)
        
        # Sample final token
        next_token = self._sample_token(aggregated, config)
        
        text = self.model_loader.decode([next_token])
        
        return GenerationResult(
            text=text,
            tokens=[next_token],
            logprobs=[],
            timing={},
            stats={}
        )
    
    def _route_experts(self, tokens: List[int]) -> Dict[int, np.ndarray]:
        """Route input to relevant experts based on gating network."""
        outputs = {}
        
        # Simple routing: distribute across all experts
        # In production, this would use learned gating weights
        num_experts = len(self.model_loader.expert_map)
        
        for expert_id in range(num_experts):
            expert_weights = self.model_loader.expert_map[expert_id]
            
            # Compute expert output (simplified)
            output = self._apply_expert(tokens, expert_weights)
            outputs[expert_id] = output
        
        return outputs
    
    def _apply_expert(
        self,
        tokens: List[int],
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply expert transformation to tokens."""
        # Simplified expert computation
        # In production, this would be actual matrix operations
        token_tensor = np.array(tokens, dtype=np.float32)
        output = np.dot(token_tensor, weights)
        return output
    
    def _sample_token(
        self,
        expert_outputs: Dict[int, np.ndarray],
        config: GenerationConfig
    ) -> int:
        """Sample next token from expert outputs."""
        # Aggregate expert outputs
        aggregated = np.mean(list(expert_outputs.values()), axis=0)
        
        # Apply temperature scaling
        scaled = aggregated / config.temperature
        
        # Softmax probability distribution
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        
        # Top-k filtering
        if config.top_k > 0:
            k_indices = np.argsort(probs)[-config.top_k:]
            probs = np.zeros_like(probs)
            probs[k_indices] = np.exp(scaled[k_indices])
            probs = probs / np.sum(probs)
        
        # Top-p filtering
        if config.top_p < 1.0:
            sorted_probs = np.sort(probs)[::-1]
            cumulative = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumulative, config.top_p)
            mask = np.zeros_like(probs)
            mask[np.argsort(sorted_probs)[:cutoff_idx + 1]] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)
        
        # Sample from distribution
        token_idx = np.random.choice(len(probs), p=probs)
        
        return int(token_idx)
    
    def _aggregate_expert_results(
        self,
        results: List[np.ndarray]
    ) -> np.ndarray:
        """Aggregate results from parallel expert computations."""
        return np.mean(results, axis=0)
    
    def _compute_expert_async(
        self,
        expert_id: int,
        tokens: List[int]
    ) -> np.ndarray:
        """Compute expert output asynchronously."""
        expert_weights = self.model_loader.expert_map[expert_id]
        return self._apply_expert(tokens, expert_weights)
    
    def _check_memory_pressure(self):
        """Check and manage memory pressure."""
        current_usage = sum(v.nbytes for v in self._memory_pool.values())
        threshold_mb = self.max_memory_mb * 0.8
        
        if current_usage / (1024 * 1024) > threshold_mb:
            logger.warning("Memory pressure detected, evicting old tensors")
            self._evict_old_tensors()
    
    def _evict_old_tensors(self):
        """Evict oldest tensors from memory pool."""
        if not self._memory_pool:
            return
        
        oldest_key = min(self._memory_pool.keys(), key=lambda k: self._memory_pool[k][0])
        del self._memory_pool[oldest_key]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "model_loaded": self._model_loaded,
            "total_tokens_generated": self.total_tokens_generated,
            "expert_stats": {
                k: {
                    "calls": v.calls,
                    "latency_ms": v.latency_ms,
                    "tokens_generated": v.tokens_generated
                }
                for k, v in self.expert_stats.items()
            },
            "memory_usage_mb": sum(v.nbytes for v in self._memory_pool.values()) / (1024 * 1024),
            "max_memory_mb": self.max_memory_mb
        }
    
    async def unload_model(self):
        """Unload model from memory."""
        if not self._model_loaded:
            return
        
        logger.info("Unloading model from memory")
        
        # Clear memory pool
        self._memory_pool.clear()
        
        # Release model resources
        await self.model_loader.unload()
        
        self._model_loaded = False
        self.total_tokens_generated = 0
    
    async def benchmark(self, prompt: str, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark model performance."""
        times = []
        tokens_per_second = []
        
        for i in range(iterations):
            start = time.time()
            result = await self.generate(
                prompt,
                GenerationConfig(max_tokens=64),
                mode=ExecutionMode.BATCHED
            )
            end = time.time()
            
            elapsed = end - start
            tps = len(result.tokens) / elapsed if elapsed > 0 else 0
            
            times.append(elapsed)
            tokens_per_second.append(tps)
        
        return {
            "avg_time_seconds": np.mean(times),
            "std_time_seconds": np.std(times),
            "avg_tokens_per_second": np.mean(tokens_per_second),
            "min_tokens_per_second": np.min(tokens_per_second),
            "max_tokens_per_second": np.max(tokens_per_second),
            "iterations": iterations
        }
</>>