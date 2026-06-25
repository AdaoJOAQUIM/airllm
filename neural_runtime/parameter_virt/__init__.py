"""Parameter Virtualization Engine - Core Module."""

from .virtualizer import ParameterVirtualizer, VirtualizationStrategy
from .sparse_gate import SparseMixtureOfExperts, ExpertRouter

__all__ = [
    "ParameterVirtualizer",
    "VirtualizationStrategy",
    "SparseMixtureOfExperts", 
    "ExpertRouter",
]
