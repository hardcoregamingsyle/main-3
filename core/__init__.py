"""
MoE Ultra Engine - Ultra-memory-efficient inference engine for Mixture-of-Experts models.

Run 2.4T parameter models like Qwen 3.8 Max on just 32GB DDR4/DDR3 RAM
at 2 seconds per token or faster.
"""

__version__ = "1.0.0"
__author__ = "Thalamus Code Project"
__email__ = "dev@thalamus.ai"
__license__ = "MIT"

from .config import Config, load_config
from .engine import MoEEngine
from .logging_utils import setup_logging, get_logger

__all__ = [
    "Config",
    "load_config",
    "MoEEngine",
    "setup_logging",
    "get_logger",
]
