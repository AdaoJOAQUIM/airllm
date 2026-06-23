"""
Neural Runtime Engine - Ultra-efficient AI inference engine
Target: 1 trillion parameter models on limited hardware
"""

__version__ = "0.1.0"
__author__ = "Neural Runtime Team"

from .core.model import NeuralRuntimeModel
from .core.inference import InferenceEngine
from .memory.hierarchy import HierarchicalMemory
from .compression.quantizer import Quantizer

__all__ = [
    "NeuralRuntimeModel",
    "InferenceEngine", 
    "HierarchicalMemory",
    "Quantizer",
]
