"""
Hierarchical Memory System
Intelligent memory management across VRAM, RAM, and SSD tiers.
"""

import os
import gc
import time
import shutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MemoryLevel(Enum):
    """Memory hierarchy levels."""
    HBM = "hbm"        # GPU VRAM - fastest
    RAM = "ram"         # System RAM
    NVME = "nvme"      # NVMe SSD
    SATA = "sata"      # SATA/HDD


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"        # Least Recently Used
    LFU = "lfu"        # Least Frequently Used
    FIFO = "fifo"      # First In First Out
    ADAPTIVE = "adaptive"  # Dynamic based on access patterns


@dataclass
class MemoryStats:
    """Memory statistics."""
    level: MemoryLevel
    used_bytes: int = 0
    capacity_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    load_time_ms: float = 0.0
    
    @property
    def usage_percent(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return (self.used_bytes / self.capacity_bytes) * 100
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


@dataclass
class LayerCache:
    """Cached layer data."""
    layer_name: str
    layer_index: int
    data: Optional[torch.Tensor] = None
    size_bytes: int = 0
    level: MemoryLevel = MemoryLevel.HBM
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    is_loading: bool = False


class HierarchicalMemory:
    """
    Hierarchical Memory System for ultra-efficient layer management.
    
    Features:
    - Multi-tier storage: HBM → RAM → NVMe → SATA
    - Intelligent prefetching based on access patterns
    - Adaptive eviction policies
    - Compressed storage for cold data
    - Non-blocking async loading
    
    Example:
        memory = HierarchicalMemory(
            max_vram_gb=24,
            max_ram_gb=128,
            storage_path="/path/to/cache"
        )
        
        # Store a layer
        memory.store("layer_0", tensor, MemoryLevel.RAM)
        
        # Retrieve a layer (auto-loads from lower tiers if needed)
        tensor = memory.get("layer_0")
    """
    
    def __init__(
        self,
        max_vram_gb: float = 24.0,
        max_ram_gb: float = 128.0,
        storage_path: Optional[Path] = None,
        cache_policy: CachePolicy = CachePolicy.ADAPTIVE,
        prefetch_workers: int = 4,
        enable_compression: bool = True,
    ):
        self.max_vram_bytes = int(max_vram_gb * 1024**3)
        self.max_ram_bytes = int(max_ram_gb * 1024**3)
        self.cache_policy = cache_policy
        self.prefetch_workers = prefetch_workers
        self.enable_compression = enable_compression
        
        # Storage path
        if storage_path:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path = Path("/tmp/neural_runtime_cache")
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Cache storage
        self._cache: Dict[MemoryLevel, OrderedDict[str, LayerCache]] = {
            MemoryLevel.HBM: OrderedDict(),
            MemoryLevel.RAM: OrderedDict(),
            MemoryLevel.NVME: OrderedDict(),
            MemoryLevel.SATA: OrderedDict(),
        }
        
        # Size tracking
        self._sizes: Dict[MemoryLevel, int] = {
            MemoryLevel.HBM: 0,
            MemoryLevel.RAM: 0,
            MemoryLevel.NVME: 0,
            MemoryLevel.SATA: 0,
        }
        
        # Statistics
        self._stats: Dict[MemoryLevel, MemoryStats] = {
            level: MemoryStats(level=level, capacity_bytes=self._get_capacity(level))
            for level in MemoryLevel
        }
        
        # Prefetch thread pool
        self._executor = ThreadPoolExecutor(max_workers=prefetch_workers)
        self._pending_futures: Dict[str, Future] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Access history for prediction
        self._access_history: List[str] = []
        self._access_patterns: Dict[str, int] = {}
        
        # Compression (placeholder)
        self._compression_available = self._check_compression()
        
        logger.info(f"HierarchicalMemory initialized:")
        logger.info(f"  HBM capacity: {max_vram_gb:.1f} GB")
        logger.info(f"  RAM capacity: {max_ram_gb:.1f} GB")
        logger.info(f"  Storage path: {self.storage_path}")
        logger.info(f"  Cache policy: {cache_policy.value}")
        logger.info(f"  Compression: {self.enable_compression and self._compression_available}")
    
    def _get_capacity(self, level: MemoryLevel) -> int:
        """Get capacity for a memory level."""
        if level == MemoryLevel.HBM:
            return self.max_vram_bytes
        elif level == MemoryLevel.RAM:
            return self.max_ram_bytes
        else:
            # Disk storage - essentially unlimited but we track usage
            return int(1000 * 1024**3)  # 1 TB assumed max
    
    def _check_compression(self) -> bool:
        """Check if compression is available."""
        try:
            import lz4
            return True
        except ImportError:
            try:
                import zstandard
                return True
            except ImportError:
                return False
    
    # ==================== Store Methods ====================
    
    def store(
        self,
        layer_name: str,
        data: torch.Tensor,
        level: MemoryLevel = MemoryLevel.HBM,
        layer_index: int = 0,
    ) -> bool:
        """
        Store a layer in the specified memory tier.
        
        Args:
            layer_name: Unique name for the layer
            data: Tensor data to store
            level: Memory tier to store in
            layer_index: Index of the layer (for ordering)
            
        Returns:
            True if successful
        """
        size_bytes = data.numel() * data.element_size()
        
        with self._lock:
            # Check if we need to evict
            if self._sizes[level] + size_bytes > self._get_capacity(level):
                evicted = self._evict_if_needed(level, size_bytes)
                if not evicted:
                    # Try lower tier
                    level = self._get_lower_tier(level)
                    if level is None:
                        logger.warning(f"Cannot store {layer_name}: all tiers full")
                        return False
            
            # Move to target level
            if self._sizes[level] + size_bytes > self._get_capacity(level):
                return False
            
            # Store data
            cache = LayerCache(
                layer_name=layer_name,
                layer_index=layer_index,
                data=data,
                size_bytes=size_bytes,
                level=level,
                last_access=time.time(),
                access_count=1,
            )
            
            # Remove from old location if exists
            self._remove_from_all_levels(layer_name)
            
            # Store
            self._cache[level][layer_name] = cache
            self._sizes[level] += size_bytes
            
            # If NVME or SATA, save to disk
            if level in [MemoryLevel.NVME, MemoryLevel.SATA]:
                self._save_to_disk(layer_name, data, level)
            
            self._record_access(layer_name)
            
            return True
    
    def _save_to_disk(self, layer_name: str, data: torch.Tensor, level: MemoryLevel) -> None:
        """Save data to disk for NVME/SATA tiers."""
        try:
            # Create directory structure
            tier_dir = self.storage_path / level.value
            tier_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as safetensor
            file_path = tier_dir / f"{layer_name}.safetensor"
            
            from safetensors.torch import save_file
            save_file({"data": data}, file_path)
            
            logger.debug(f"Saved {layer_name} to {level.value} disk")
        except Exception as e:
            logger.error(f"Failed to save {layer_name} to disk: {e}")
    
    # ==================== Retrieve Methods ====================
    
    def get(
        self,
        layer_name: str,
        target_level: MemoryLevel = MemoryLevel.HBM,
        async_load: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a layer from memory.
        
        Automatically handles loading from lower tiers if needed.
        
        Args:
            layer_name: Name of the layer
            target_level: Desired memory tier to load into
            async_load: If True, return immediately if loading in background
            
        Returns:
            Tensor data or None if not found
        """
        with self._lock:
            # Check if already in memory
            for level in MemoryLevel:
                if layer_name in self._cache[level]:
                    cache = self._cache[level][layer_name]
                    cache.last_access = time.time()
                    cache.access_count += 1
                    self._stats[level].hits += 1
                    self._record_access(layer_name)
                    
                    # Promote to target level if needed
                    if level != target_level and target_level == MemoryLevel.HBM:
                        self._promote(layer_name, cache, target_level)
                    
                    return cache.data
            
            # Not in cache - check if loading in background
            if layer_name in self._pending_futures:
                if async_load:
                    return None
                future = self._pending_futures[layer_name]
                if future.done():
                    return future.result()
                else:
                    # Wait for load
                    return future.result()
            
            # Load from disk
            self._stats[target_level].misses += 1
            
            # Try to load from any available tier
            data = self._load_from_disk(layer_name)
            
            if data is not None:
                # Store in target level
                self.store(layer_name, data, target_level)
                return data
            
            return None
    
    def _load_from_disk(self, layer_name: str) -> Optional[torch.Tensor]:
        """Load layer from disk."""
        from safetensors.torch import load_file
        
        for level in [MemoryLevel.NVME, MemoryLevel.SATA]:
            file_path = self.storage_path / level.value / f"{layer_name}.safetensor"
            if file_path.exists():
                try:
                    data = load_file(file_path)["data"]
                    return data
                except Exception as e:
                    logger.error(f"Failed to load {layer_name} from {level.value}: {e}")
        
        return None
    
    def _promote(self, layer_name: str, cache: LayerCache, target_level: MemoryLevel) -> None:
        """Promote a layer to a higher memory tier."""
        if cache.data is None:
            return
        
        # Check capacity
        if self._sizes[target_level] + cache.size_bytes > self._get_capacity(target_level):
            self._evict_if_needed(target_level, cache.size_bytes)
        
        # Move data
        self._cache[cache.level].pop(layer_name, None)
        self._sizes[cache.level] -= cache.size_bytes
        
        cache.level = target_level
        cache.last_access = time.time()
        
        self._cache[target_level][layer_name] = cache
        self._sizes[target_level] += cache.size_bytes
    
    # ==================== Prefetch Methods ====================
    
    def prefetch(
        self,
        layer_names: List[str],
        target_level: MemoryLevel = MemoryLevel.HBM,
    ) -> Dict[str, Future]:
        """
        Prefetch multiple layers asynchronously.
        
        Args:
            layer_names: List of layer names to prefetch
            target_level: Target memory tier
            
        Returns:
            Dictionary mapping layer names to futures
        """
        futures = {}
        
        for layer_name in layer_names:
            if layer_name in self._pending_futures:
                futures[layer_name] = self._pending_futures[layer_name]
            elif layer_name not in self._cache[target_level]:
                # Submit async load
                future = self._executor.submit(self._async_load, layer_name, target_level)
                self._pending_futures[layer_name] = future
                futures[layer_name] = future
        
        return futures
    
    def _async_load(self, layer_name: str, target_level: MemoryLevel) -> Optional[torch.Tensor]:
        """Async load a layer from disk."""
        data = self._load_from_disk(layer_name)
        if data is not None:
            self.store(layer_name, data, target_level)
        return data
    
    def wait_for_prefetch(self, layer_name: str, timeout: float = 30.0) -> Optional[torch.Tensor]:
        """Wait for a prefetch to complete."""
        if layer_name not in self._pending_futures:
            return self.get(layer_name)
        
        future = self._pending_futures[layer_name]
        try:
            result = future.result(timeout=timeout)
            self._pending_futures.pop(layer_name, None)
            return result
        except TimeoutError:
            logger.warning(f"Prefetch for {layer_name} timed out")
            return None
    
    # ==================== Eviction Methods ====================
    
    def _evict_if_needed(self, level: MemoryLevel, needed_bytes: int) -> bool:
        """Evict layers to make room for new data."""
        while self._sizes[level] + needed_bytes > self._get_capacity(level):
            if not self._cache[level]:
                return False
            
            if self.cache_policy == CachePolicy.LRU:
                self._evict_lru(level)
            elif self.cache_policy == CachePolicy.LFU:
                self._evict_lfu(level)
            elif self.cache_policy == CachePolicy.FIFO:
                self._evict_fifo(level)
            elif self.cache_policy == CachePolicy.ADAPTIVE:
                self._evict_adaptive(level)
            else:
                self._evict_lru(level)
        
        return True
    
    def _evict_lru(self, level: MemoryLevel) -> None:
        """Evict Least Recently Used layer."""
        if not self._cache[level]:
            return
        
        # Remove oldest
        layer_name, cache = self._cache[level].popitem(last=False)
        self._sizes[level] -= cache.size_bytes
        self._stats[level].evictions += 1
        
        # Demote to lower tier if possible
        lower_level = self._get_lower_tier(level)
        if lower_level and cache.data is not None:
            self.store(layer_name, cache.data, lower_level, cache.layer_index)
        
        logger.debug(f"Evicted {layer_name} from {level.value} (LRU)")
    
    def _evict_lfu(self, level: MemoryLevel) -> None:
        """Evict Least Frequently Used layer."""
        if not self._cache[level]:
            return
        
        # Find layer with lowest access count
        lfu_name = min(
            self._cache[level].keys(),
            key=lambda x: self._cache[level][x].access_count
        )
        
        cache = self._cache[level].pop(lfu_name)
        self._sizes[level] -= cache.size_bytes
        self._stats[level].evictions += 1
        
        lower_level = self._get_lower_tier(level)
        if lower_level and cache.data is not None:
            self.store(lfu_name, cache.data, lower_level, cache.layer_index)
    
    def _evict_fifo(self, level: MemoryLevel) -> None:
        """Evict First In First Out."""
        if self._cache[level]:
            layer_name, cache = self._cache[level].popitem(last=False)
            self._sizes[level] -= cache.size_bytes
            self._stats[level].evictions += 1
    
    def _evict_adaptive(self, level: MemoryLevel) -> None:
        """Adaptive eviction based on multiple factors."""
        if not self._cache[level]:
            return
        
        # Score = age * (1 / access_count) * size
        # Lower score = better candidate for eviction
        current_time = time.time()
        
        best_candidate = None
        best_score = float('inf')
        
        for name, cache in self._cache[level].items():
            age = current_time - cache.last_access
            score = age * (1.0 / max(cache.access_count, 1))
            
            # Larger layers are better candidates
            score *= (cache.size_bytes / (1024**2))  # Size in MB
            
            if score < best_score:
                best_score = score
                best_candidate = name
        
        if best_candidate:
            cache = self._cache[level].pop(best_candidate)
            self._sizes[level] -= cache.size_bytes
            self._stats[level].evictions += 1
            
            lower_level = self._get_lower_tier(level)
            if lower_level and cache.data is not None:
                self.store(best_candidate, cache.data, lower_level, cache.layer_index)
    
    def _get_lower_tier(self, level: MemoryLevel) -> Optional[MemoryLevel]:
        """Get the next lower memory tier."""
        tier_order = [MemoryLevel.HBM, MemoryLevel.RAM, MemoryLevel.NVME, MemoryLevel.SATA]
        try:
            idx = tier_order.index(level)
            if idx < len(tier_order) - 1:
                return tier_order[idx + 1]
        except ValueError:
            pass
        return None
    
    def _remove_from_all_levels(self, layer_name: str) -> None:
        """Remove a layer from all memory tiers."""
        for level in MemoryLevel:
            if layer_name in self._cache[level]:
                cache = self._cache[level].pop(layer_name)
                self._sizes[level] -= cache.size_bytes
    
    # ==================== Access Pattern Tracking ====================
    
    def _record_access(self, layer_name: str) -> None:
        """Record layer access for pattern prediction."""
        self._access_history.append(layer_name)
        self._access_patterns[layer_name] = self._access_patterns.get(layer_name, 0) + 1
        
        # Keep history bounded
        if len(self._access_history) > 10000:
            self._access_history = self._access_history[-5000:]
    
    def predict_next_layers(self, current_layer: int, count: int = 3) -> List[str]:
        """
        Predict which layers will be accessed next based on access patterns.
        
        Args:
            current_layer: Current layer index
            count: Number of predictions
            
        Returns:
            List of predicted layer names
        """
        # Simple pattern: sequential access is most common
        predictions = []
        
        for i in range(1, count + 1):
            next_layer = current_layer + i
            if next_layer < len(self._cache.get(MemoryLevel.HBM, {})):
                predictions.append(f"layer_{next_layer}")
        
        # If we don't have enough, use history
        if len(predictions) < count and self._access_history:
            recent = self._access_history[-count * 2:]
            for name in reversed(recent):
                if name not in predictions:
                    predictions.append(name)
                    if len(predictions) >= count:
                        break
        
        return predictions
    
    # ==================== Utility Methods ====================
    
    def clear(self, level: Optional[MemoryLevel] = None) -> None:
        """Clear memory cache."""
        with self._lock:
            if level:
                self._cache[level].clear()
                self._sizes[level] = 0
            else:
                for lvl in MemoryLevel:
                    self._cache[lvl].clear()
                    self._sizes[lvl] = 0
            
            gc.collect()
    
    def get_stats(self) -> Dict[MemoryLevel, MemoryStats]:
        """Get memory statistics."""
        return self._stats.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of memory usage."""
        summary = {}
        
        for level in MemoryLevel:
            stats = self._stats[level]
            cache = self._cache[level]
            
            summary[level.value] = {
                "used_gb": self._sizes[level] / (1024**3),
                "capacity_gb": self._get_capacity(level) / (1024**3),
                "usage_percent": stats.usage_percent,
                "items_cached": len(cache),
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions,
            }
        
        return summary
    
    def __del__(self):
        """Cleanup on deletion."""
        self._executor.shutdown(wait=False)
