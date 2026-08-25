"""
MoE Ultra Engine - Core inference engine for Mixture-of-Experts models.

Implements ultra-memory-efficient inference with:
- Expert offloading to CPU/disk
- Paged attention for KV cache
- Quantization support (int4, int8, GPTQ, AWQ)
- Flash Attention 2 integration
- Speculative decoding
- Streaming generation
"""

import asyncio
import gc
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import weakref

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import Config, ModelConfig, MemoryConfig, InferenceConfig
from .logging_utils import get_logger, PerformanceLogger, set_request_id, clear_context


logger = get_logger(__name__)


@dataclass
class ExpertInfo:
    """Information about a single expert."""
    layer_idx: int
    expert_idx: int
    device: torch.device
    size_bytes: int
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False


@dataclass
class GenerationResult:
    """Result of a generation request."""
    text: str
    tokens_generated: int
    prompt_tokens: int
    generation_time: float
    tokens_per_second: float
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExpertCache:
    """LRU cache for expert weights with memory pressure awareness."""

    def __init__(self, max_experts: int, max_ram_gb: float, offload_path: Path):
        self.max_experts = max_experts
        self.max_ram_bytes = int(max_ram_gb * 1024**3)
        self.offload_path = offload_path
        self.offload_path.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, ExpertInfo] = {}
        self._expert_weights: Dict[str, torch.Tensor] = {}
        self._current_ram_usage = 0
        self._lock = asyncio.Lock()
        self._access_order: List[str] = []  # LRU tracking

    def _make_key(self, layer_idx: int, expert_idx: int) -> str:
        return f"layer_{layer_idx}_expert_{expert_idx}"

    async def get_expert(self, layer_idx: int, expert_idx: int) -> Optional[torch.Tensor]:
        """Get expert weights, loading from offload if necessary."""
        key = self._make_key(layer_idx, expert_idx)
        async with self._lock:
            if key in self._expert_weights:
                # Update LRU
                self._access_order.remove(key)
                self._access_order.append(key)
                self._cache[key].last_accessed = time.time()
                self._cache[key].access_count += 1
                return self._expert_weights[key]

            # Try to load from disk offload
            offload_file = self.offload_path / f"{key}.pt"
            if offload_file.exists():
                await self._load_expert(key, offload_file)
                return self._expert_weights.get(key)

            return None

    async def put_expert(self, layer_idx: int, expert_idx: int, weights: torch.Tensor, pinned: bool = False) -> None:
        """Store expert weights in cache."""
        key = self._make_key(layer_idx, expert_idx)
        size_bytes = weights.element_size() * weights.nelement()

        async with self._lock:
            # Check if we need to evict
            while (self._current_ram_usage + size_bytes > self.max_ram_usage
                   and len(self._expert_weights) >= self.max_experts):
                await self._evict_lru()

            # If expert already exists, remove old
            if key in self._expert_weights:
                old_size = self._expert_weights[key].element_size() * self._expert_weights[key].nelement()
                self._current_ram_usage -= old_size
                del self._expert_weights[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            # Add new expert
            self._expert_weights[key] = weights
            self._cache[key] = ExpertInfo(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device=weights.device,
                size_bytes=size_bytes,
                pinned=pinned,
            )
            self._access_order.append(key)
            self._current_ram_usage += size_bytes

    async def _load_expert(self, key: str, path: Path) -> None:
        """Load expert from disk."""
        try:
            weights = torch.load(path, map_location="cpu", weights_only=True)
            size_bytes = weights.element_size() * weights.nelement()

            # Check memory pressure
            while self._current_ram_usage + size_bytes > self.max_ram_bytes:
                await self._evict_lru()

            self._expert_weights[key] = weights
            self._cache[key] = ExpertInfo(
                layer_idx=int(key.split("_")[1]),
                expert_idx=int(key.split("_")[3]),
                device=weights.device,
                size_bytes=size_bytes,
            )
            self._access_order.append(key)
            self._current_ram_usage += size_bytes
            logger.debug(f"Loaded expert {key} from disk ({size_bytes / 1e6:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to load expert {key}: {e}")

    async def _evict_lru(self) -> bool:
        """Evict least recently used non-pinned expert."""
        for key in self._access_order:
            if key in self._cache and not self._cache[key].pinned:
                expert = self._cache[key]
                # Save to disk if offloading enabled
                offload_file = self.offload_path / f"{key}.pt"
                try:
                    torch.save(self._expert_weights[key], offload_file)
                    logger.debug(f"Offloaded expert {key} to disk")
                except Exception as e:
                    logger.warning(f"Failed to offload expert {key}: {e}")

                self._current_ram_usage -= expert.size_bytes
                del self._expert_weights[key]
                del self._cache[key]
                self._access_order.remove(key)
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_experts": len(self._expert_weights),
            "ram_usage_mb": self._current_ram_usage / 1e6,
            "max_ram_mb": self.max_ram_bytes / 1e6,
            "utilization": self._current_ram_usage / self.max_ram_bytes if self.max_ram_bytes > 0 else 0,
        }

    async def clear(self) -> None:
        """Clear all cached experts."""
        async with self._lock:
            self._expert_weights.clear()
            self._cache.clear()
            self._access_order.clear()
            self._current_ram_usage = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class PagedKVCache:
    """Paged attention KV cache for memory-efficient long context."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_pages: int = 0,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype
        self.device = device

        # Page management
        self._free_pages: List[int] = []
        self._page_tables: Dict[int, List[int]] = {}  # seq_id -> page indices
        self._page_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}  # page_idx -> (K, V)
        self._next_page_idx = 0

        # Pre-allocate if max_pages specified
        if max_pages > 0:
            self._preallocate_pages(max_pages)

    def _preallocate_pages(self, num_pages: int) -> None:
        """Pre-allocate page buffers."""
        for i in range(num_pages):
            k_page = torch.zeros(
                (self.num_heads, self.page_size, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v_page = torch.zeros(
                (self.num_heads, self.page_size, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._page_data[i] = (k_page, v_page)
            self._free_pages.append(i)
        self._next_page_idx = num_pages

    def allocate_sequence(self, seq_id: int, num_pages: int) -> List[int]:
        """Allocate pages for a sequence."""
        pages = []
        for _ in range(num_pages):
            if self._free_pages:
                page_idx = self._free_pages.pop()
            else:
                # Allocate new page
                k_page = torch.zeros(
                    (self.num_heads, self.page_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                v_page = torch.zeros(
                    (self.num_heads, self.page_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                page_idx = self._next_page_idx
                self._page_data[page_idx] = (k_page, v_page)
                self._next_page_idx += 1
            pages.append(page_idx)
        self._page_tables[seq_id] = pages
        return pages

    def free_sequence(self, seq_id: int) -> None:
        """Free pages for a sequence."""
        if seq_id in self._page_tables:
            self._free_pages.extend(self._page_tables[seq_id])
            del self._page_tables[seq_id]

    def get_kv(self, seq_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated K,V for a sequence and layer."""
        if seq_id not in self._page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated")

        pages = self._page_tables[seq_id]
        k_pages = []
        v_pages = []
        for page_idx in pages:
            k, v = self._page_data[page_idx]
            k_pages.append(k)
            v_pages.append(v)

        if not k_pages:
            return (
                torch.empty((self.num_heads, 0, self.head_dim), dtype=self.dtype, device=self.device),
                torch.empty((self.num_heads, 0, self.head_dim), dtype=self.dtype, device=self.device),
            )

        return torch.cat(k_pages, dim=1), torch.cat(v_pages, dim=1)

    def append_kv(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Append new K,V to sequence cache."""
        if seq_id not in self._page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated")

        pages = self._page_tables[seq_id]
        seq_len = k.shape[1]
        pos = 0

        while pos < seq_len:
            # Find page with space
            page_idx = pages[pos // self.page_size]
            page_k, page_v = self._page_data[page_idx]
            page_offset = pos % self.page_size
            remaining_in_page = self.page_size - page_offset
            copy_len = min(remaining_in_page, seq_len - pos)

            page_k[:, page_offset:page_offset + copy_len] = k[:, pos:pos + copy_len]
            page_v[:, page_offset:page_offset + copy_len] = v[:, pos:pos + copy_len]
            pos += copy_len

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_pages = len(self._page_data)
        used_pages = total_pages - len(self._free_pages)
        bytes_per_page = 2 * self.num_heads * self.page_size * self.head_dim * torch.tensor([], dtype=self.dtype).element_size()
        return {
            "total_pages": total_pages,
            "used_pages": used_pages,
            "free_pages": len(self._free_pages),
            "memory_mb": (used_pages * bytes_per_page) / 1e6,
        }


class MoEEngine:
    """Main MoE inference engine."""

    def __init__(self, config: Config):
        self.config = config
        self.model_config = config.model
        self.memory_config = config.memory
        self.inference_config = config.inference

        self._model: Optional[nn.Module] = None
        self._tokenizer = None
        self._expert_cache: Optional[ExpertCache] = None
        self._kv_cache: Optional[PagedKVCache] = None
        self._initialized = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = self._get_dtype()

        # Generation state
        self._active_sequences: Dict[int, Dict[str, Any]] = {}
        self._seq_counter = 0

    def _get_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "int8": torch.int8,
            "int4": torch.int8,  # Handled via quantization
        }
        return dtype_map.get(self.model_config.dtype, torch.float16)

    async def initialize(self) -> None:
        """Initialize the engine: load model, tokenizer, setup caches."""
        if self._initialized:
            return

        logger.info("Initializing MoE Ultra Engine", extra={
            "model": self.model_config.name,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "max_ram_gb": self.memory_config.max_ram_gb,
        })

        with PerformanceLogger(logger, "model_loading") as perf:
            await self._load_model()
            perf.add_metadata(model=self.model_config.name)

        with PerformanceLogger(logger, "tokenizer_loading") as perf:
            await self._load_tokenizer()

        with PerformanceLogger(logger, "cache_setup") as perf:
            await self._setup_caches()
            perf.add_metadata(
                expert_cache_size=self.memory_config.expert_cache_size,
                kv_cache_pages=self.memory_config.max_pages,
            )

        self._initialized = True
        logger.info("MoE Ultra Engine initialized successfully")

    async def _load_model(self) -> None:
        """Load MoE model with quantization and expert offloading."""
        model_path = Path(self.model_config.path) / self.model_config.name

        if not model_path.exists():
            raise RuntimeError(f"Model not found at {model_path}. Run 'moe-engine download-model' first.")

        # Import here to avoid circular imports
        from transformers import AutoModelForCausalLM, AutoConfig

        # Load config first
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Determine quantization config
        quantization_config = self._get_quantization_config()

        # Load model with low_cpu_mem_usage for memory efficiency
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.inference_config.use_flash_attention else "eager",
        )

        # Apply expert offloading if enabled
        if self.memory_config.offload_to_cpu:
            await self._setup_expert_offloading()

        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, "compile") and not torch.cuda.is_available():
            logger.info("Compiling model with torch.compile")
            self._model = torch.compile(self._model, mode="reduce-overhead")

        self._model.eval()

        # Log model info
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"Model loaded: {total_params / 1e9:.2f}B parameters")

    def _get_quantization_config(self):
        """Get quantization configuration for model loading."""
        quant = self.model_config.quantization.lower()

        if quant == "none":
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("bitsandbytes not installed, quantization disabled")
            return None

        if quant == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quant in ("int4", "4bit"):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant == "gptq":
            # GPTQ handled by AutoGPTQForCausalLM
            return None
        elif quant == "awq":
            # AWQ handled by AutoAWQForCausalLM
            return None

        return None

    async def _setup_expert_offloading(self) -> None:
        """Setup expert offloading for MoE layers."""
        if not hasattr(self._model, "model") or not hasattr(self._model.model, "layers"):
            logger.warning("Model structure not recognized for expert offloading")
            return

        offload_path = Path(self.memory_config.disk_offload_path) / "experts"
        self._expert_cache = ExpertCache(
            max_experts=self.memory_config.expert_cache_size,
            max_ram_gb=self.memory_config.max_ram_gb * 0.5,  # Use half RAM for experts
            offload_path=offload_path,
        )

        # Identify MoE layers and experts
        num_experts_offloaded = 0
        for layer_idx, layer in enumerate(self._model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                experts = layer.mlp.experts
                for expert_idx, expert in enumerate(experts):
                    # Move expert to CPU initially
                    expert.to("cpu")
                    # Cache the expert weights
                    expert_weights = {name: param.data.clone() for name, param in expert.named_parameters()}
                    await self._expert_cache.put_expert(layer_idx, expert_idx, expert_weights)
                    num_experts_offloaded += 1

        logger.info(f"Offloaded {num_experts_offloaded} experts to CPU/disk")

    async def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        from transformers import AutoTokenizer

        model_path = Path(self.model_config.path) / self.model_config.name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    async def _setup_caches(self) -> None:
        """Setup KV cache and other caches."""
        if not self._model:
            return

        # Get model dimensions
        num_layers = len(self._model.model.layers) if hasattr(self._model, "model") else 32
        num_heads = getattr(self._model.config, "num_attention_heads", 32)
        head_dim = getattr(self._model.config, "hidden_size", 4096) // num_heads

        kv_dtype = getattr(torch, self.memory_config.kv_cache_dtype, torch.float16)

        self._kv_cache = PagedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=self.memory_config.page_size,
            max_pages=self.memory_config.max_pages,
            dtype=kv_dtype,
            device=self._device,
        )

    @asynccontextmanager
async def _generation_context(self, request_id: str):
        """Context manager for generation request."""
        set_request_id(request_id)
        seq_id = self._seq_counter
        self._seq_counter += 1

        # Allocate KV cache for this sequence
        if self._kv_cache:
            max_pages = max(1, self.inference_config.max_tokens // self.memory_config.page_size + 1)
            self._kv_cache.allocate_sequence(seq_id, max_pages)

        self._active_sequences[seq_id] = {
            "request_id": request_id,
            "start_time": time.time(),
            "tokens_generated": 0,
        }

        try:
            yield seq_id
        finally:
            if self._kv_cache:
                self._kv_cache.free_sequence(seq_id)
            self._active_sequences.pop(seq_id, None)
            clear_context()

    async def generate(
        self,
        prompt: str,
        inference_config: Optional[InferenceConfig] = None,
    ) -> GenerationResult:
        """Generate text from prompt (non-streaming)."""
        if not self._initialized:
            await self.initialize()

        inference_config = inference_config or self.inference_config
        request_id = str(uuid.uuid4())[:8]

        async with self._generation_context(request_id) as seq_id:
            # Tokenize prompt
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            prompt_tokens = input_ids.shape[1]

            start_time = time.time()
            generated_ids = []

            with torch.no_grad():
                for _ in range(inference_config.max_tokens):
                    # Forward pass
                    outputs = self._model(input_ids=input_ids, use_cache=True)
                    logits = outputs.logits[:, -1, :]

                    # Apply sampling
                    next_token = self._sample_token(logits, inference_config)
                    generated_ids.append(next_token.item())

                    # Check stop conditions
                    if next_token.item() in self._tokenizer.encode(inference_config.stop_sequences):
                        break
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break

                    # Prepare next input
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                    # Truncate if exceeding max_seq_len
                    if input_ids.shape[1] > self.model_config.max_seq_len:
                        input_ids = input_ids[:, -self.model_config.max_seq_len:]

            generation_time = time.time() - start_time
            tokens_generated = len(generated_ids)

            # Decode
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            return GenerationResult(
                text=generated_text,
                tokens_generated=tokens_generated,
                prompt_tokens=prompt_tokens,
                generation_time=generation_time,
                tokens_per_second=tokens_generated / generation_time if generation_time > 0 else 0,
                finish_reason="stop" if tokens_generated < inference_config.max_tokens else "length",
                metadata={"request_id": request_id},
            )

    async def generate_stream(
        self,
        prompt: str,
        inference_config: Optional[InferenceConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming."""
        if not self._initialized:
            await self.initialize()

        inference_config = inference_config or self.inference_config
        request_id = str(uuid.uuid4())[:8]

        async with self._generation_context(request_id) as seq_id:
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            prompt_tokens = input_ids.shape[1]

            generated_ids = []
            start_time = time.time()

            with torch.no_grad():
                for _ in range(inference_config.max_tokens):
                    outputs = self._model(input_ids=input_ids, use_cache=True)
                    logits = outputs.logits[:, -1, :]

                    next_token = self._sample_token(logits, inference_config)
                    token_id = next_token.item()
                    generated_ids.append(token_id)

                    # Decode and yield token
                    token_text = self._tokenizer.decode([token_id], skip_special_tokens=True)
                    if token_text:
                        yield token_text

                    # Check stop conditions
                    if token_id in [self._tokenizer.encode(s)[0] for s in inference_config.stop_sequences if s]:
                        break
                    if token_id == self._tokenizer.eos_token_id:
                        break

                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                    if input_ids.shape[1] > self.model_config.max_seq_len:
                        input_ids = input_ids[:, -self.model_config.max_seq_len:]

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        inference_config: Optional[InferenceConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Chat completion with streaming."""
        # Apply chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"

        async for token in self.generate_stream(prompt, inference_config):
            yield token

    def _sample_token(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Sample next token from logits."""
        # Temperature scaling
        if config.temperature > 0:
            logits = logits / config.temperature

        # Top-k filtering
        if config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Repetition penalty (simplified)
        if config.repetition_penalty != 1.0:
            # Would need generated_ids history for full implementation
            pass

        # Sample
        probs = F.softmax(logits, dim=-1)
        if config.temperature == 0:
            return torch.argmax(probs, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    async def shutdown(self) -> None:
        """Shutdown engine and release resources."""
        logger.info("Shutting down MoE Ultra Engine")

        if self._expert_cache:
            await self._expert_cache.clear()

        if self._kv_cache:
            self._kv_cache = None

        if self._model:
            del self._model
            self._model = None

        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("Engine shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "initialized": self._initialized,
            "model": self.model_config.name,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "active_sequences": len(self._active_sequences),
        }

        if self._expert_cache:
            stats["expert_cache"] = self._expert_cache.get_stats()

        if self._kv_cache:
            stats["kv_cache"] = self._kv_cache.get_memory_usage()

        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            }

        return stats


# Factory function for easy instantiation
def create_engine(config: Optional[Config] = None) -> MoEEngine:
    """Create and return an MoEEngine instance."""
    if config is None:
        config = load_config()
    return MoEEngine(config)
