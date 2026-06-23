"""Cache module - Intelligent caching."""

from .predictor import AccessPatternPredictor
from .lru import LRUCache

__all__ = ["AccessPatternPredictor", "LRUCache"]
