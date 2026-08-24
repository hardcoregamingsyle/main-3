"""
MoE Ultra Engine - Core Inference Engine

Ultra-memory-efficient MoE inference for running 2.4T parameter models
on consumer hardware with 32GB RAM.
"""

__version__ = "1.0.0"
__author__ = "Thalamus Code Project"
__license__ = "MIT"

from .loader import ModelLoader
from .offloader import OffloadManager
from .scheduler import LayerScheduler
from .quantizer import Quantizer
from .cache import KVCacheManager
from .metrics import MetricsCollector

__all__ = [
    "ModelLoader",
    "OffloadManager",
    "LayerScheduler",
    "Quantizer",
    "KVCacheManager",
    "MetricsCollector",
]
