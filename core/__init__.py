"""
MoE Ultra Engine - Ultra-memory-efficient inference engine for Mixture-of-Experts models.
"""

__version__ = "1.0.0"
__author__ = "Thalamus Code Project"
__email__ = "dev@thalamus.ai"
__license__ = "MIT"

from .config import MoEConfig, EngineConfig, ExpertConfig, MemoryConfig
from .engine import MoEEngine

__all__ = [
    "MoEConfig",
    "EngineConfig",
    "ExpertConfig",
    "MemoryConfig",
    "MoEEngine",
]
