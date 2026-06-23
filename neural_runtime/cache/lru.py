"""
LRU Cache implementation for layer caching.
"""

from typing import Any, Optional, Dict, Callable
from collections import OrderedDict
import threading
import time
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Thread-safe LRU Cache with size limits.
    
    Features:
    - Size limits (items or bytes)
    - Automatic eviction
    - Thread-safe operations
    - Statistics tracking
    
    Example:
        cache = LRUCache(max_items=100, max_size_mb=1000)
        cache.put("layer_0", tensor)
        tensor = cache.get("layer_0")
    """
    
    def __init__(
        self,
        max_items: Optional[int] = None,
        max_size_mb: Optional[float] = None,
        on_evict: Optional[Callable] = None,
    ):
        self.max_items = max_items
        self.max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None
        self.on_evict = on_evict
        
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_size = 0
        
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        """Put item in cache."""
        with self._lock:
            # Check if key exists
            if key in self._cache:
                # Update existing
                old_size = self._sizes.get(key, 0)
                self._total_size -= old_size
                self._cache.move_to_end(key)
            
            # Evict if needed
            while self._should_evict(size_bytes):
                self._evict_one()
            
            # Add new item
            self._cache[key] = value
            self._sizes[key] = size_bytes
            self._total_size += size_bytes
    
    def _should_evict(self, new_item_size: int = 0) -> bool:
        """Check if eviction is needed."""
        if self.max_items and len(self._cache) >= self.max_items:
            return True
        
        if self.max_size_bytes and self._total_size + new_item_size > self.max_size_bytes:
            return True
        
        return False
    
    def _evict_one(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Pop oldest (first) item
        key, value = self._cache.popitem(last=False)
        size = self._sizes.pop(key, 0)
        self._total_size -= size
        self.evictions += 1
        
        # Call eviction callback
        if self.on_evict:
            try:
                self.on_evict(key, value)
            except Exception as e:
                logger.error(f"Error in eviction callback: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                size = self._sizes.pop(key, 0)
                self._total_size -= size
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._total_size = 0
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Get number of items."""
        return len(self._cache)
    
    @property
    def size_bytes(self) -> int:
        """Get total size in bytes."""
        return self._total_size
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "items": len(self._cache),
            "total_size_mb": self._total_size / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }
