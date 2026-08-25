"""
MoE Ultra Engine - Ultra-memory-efficient inference engine for Mixture-of-Experts models.
"""

__version__ = "1.0.0"
__author__ = "Thalamus Code Project"
__email__ = "dev@thalamus.ai"

from .engine import MoEEngine
from .config import EngineConfig, ModelConfig, QuantizationConfig

__all__ = [
    "MoEEngine",
    "EngineConfig",
    "ModelConfig",
    "QuantizationConfig",
]
