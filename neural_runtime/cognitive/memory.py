"""
Cognitive Memory Hierarchy
======================

A proactive memory system that predicts and prefetches based on access patterns.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory tier hierarchy - from fastest to slowest."""
    ULTRA_CACHE = "ultra_cache"  # GPU registers/specialized cache
    HBM = "hbm"                  # GPU VRAM
    RAM = "ram"                  # System RAM
    SSD = "ssd"                  # NVMe/SATA SSD
    GENERATED = "generated"      # On-demand generation


@dataclass
class MemoryStats:
    """Statistics for memory tier."""
    tier: MemoryTier
    size_bytes: int
    used_bytes: int = 0
    hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0
    
    @property
    def usage_percent(self) -> float:
        return (self.used_bytes / max(self.size_bytes, 1)) * 100
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


class CognitiveMemory:
    """
    Cognitive Memory System with proactive prediction.
    
    Unlike traditional memory that reacts to requests, this memory:
    1. Tracks access patterns
    2. Predicts future needs
    3. Prefetches before requested
    4. Adapts tier placement based on patterns
    
    ┌────────────────────────────────────────────────────────────────┐
    │                   COGNITIVE MEMORY FLOW                          │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   Query ──► Predictor ──► Prefetch ──► [Active when needed]    │
    │                  ▲                                               │
    │                  │                                               │
    │             Pattern Learning                                     │
    │                  │                                               │
    │             Access History                                       │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        ultra_cache_size_mb: int = 512,
        vram_size_gb: float = 24.0,
        ram_size_gb: float = 128.0,
        ssd_path: Optional[str] = None,
    ):
        # Tier sizes
        self.tier_sizes = {
            MemoryTier.ULTRA_CACHE: ultra_cache_size_mb * 1024**2,
            MemoryTier.HBM: int(vram_size_gb * 1024**3),
            MemoryTier.RAM: int(ram_size_gb * 1024**3),
            MemoryTier.SSD: 1_000_000 * 1024**2,  # 1 TB assumed
            MemoryTier.GENERATED: float('inf'),  # No limit
        }
        
        # Tier storage
        self.tier_data: Dict[MemoryTier, Dict[str, Any]] = {
            tier: {} for tier in MemoryTier
        }
        
        # Statistics
        self.stats: Dict[MemoryTier, MemoryStats] = {
            tier: MemoryStats(
                tier=tier,
                size_bytes=self.tier_sizes[tier]
            )
            for tier in MemoryTier
        }
        
        # Access history
        self.access_history: List[str] = []
        self.access_timestamps: Dict[str, List[float]] = {}
        
        # Pattern tracker
        self.sequential_score: float = 0.0
        self.layer_transitions: Dict[Tuple[str, str], int] = {}
        
        # Prediction model (simple)
        self.last_n_accesses: int = 10
        
        # SSD path
        self.ssd_path = ssd_path
        
        logger.info("CognitiveMemory initialized:")
        for tier in MemoryTier:
            size_mb = self.tier_sizes[tier] / (1024**2)
            logger.info(f"  {tier.value}: {size_mb:.0f} MB")
    
    def store(
        self,
        key: str,
        data: Any,
        tier: MemoryTier = MemoryTier.HBM,
        predicted: bool = False,
    ) -> bool:
        """
        Store data in a memory tier.
        
        Args:
            key: Unique identifier for the data
            data: Data to store
            tier: Target memory tier
            predicted: Whether this was prefetched (for tracking)
        """
        # Track access
        self._record_access(key, predicted)
        
        # Store
        self.tier_data[tier][key] = data
        size = self._estimate_size(data)
        
        # Update stats
        self.stats[tier].used_bytes += size
        if predicted:
            self.stats[tier].prefetch_hits += 1
        
        # Evict if needed
        self._auto_evict(tier)
        
        return True
    
    def get(
        self,
        key: str,
        target_tier: MemoryTier = MemoryTier.HBM,
    ) -> Optional[Any]:
        """
        Get data from memory, potentially promoting it.
        
        Args:
            key: Data identifier
            target_tier: Preferred tier to retrieve from
            
        Returns:
            Data if found, None otherwise
        """
        # Check tiers in order
        for tier in MemoryTier:
            if key in self.tier_data[tier]:
                # Hit!
                self.stats[tier].hits += 1
                self._record_access(key)
                
                # Promote to target tier if higher priority
                if tier != target_tier:
                    if self._tier_priority(tier) > self._tier_priority(target_tier):
                        self._promote(key, tier, target_tier)
                
                return self.tier_data[tier][key]
        
        # Miss
        for tier in MemoryTier:
            self.stats[tier].misses += 1
        
        return None
    
    def predict_and_prefetch(
        self,
        current_key: str,
        count: int = 3,
    ) -> List[str]:
        """
        Predict next keys and prefetch them.
        
        Args:
            current_key: Current key being accessed
            count: Number of predictions
            
        Returns:
            List of predicted keys
        """
        predictions = self._predict_next(current_key, count)
        
        # Prefetch predictions
        prefetched = []
        for key in predictions:
            if not self.contains(key):
                # Try to prefetch from SSD or generate
                if key in self.tier_data.get(MemoryTier.SSD, {}):
                    self.tier_data[MemoryTier.SSD][key]
                    self.store(key, self.tier_data[MemoryTier.SSD].get(key), 
                            MemoryTier.RAM, predicted=True)
                prefetched.append(key)
        
        return predictions
    
    def _predict_next(self, current_key: str, count: int) -> List[str]:
        """
        Predict the next keys to be accessed.
        
        Uses multiple strategies:
        1. Sequential prediction (most common)
        2. Transition-based prediction
        3. Frequency-based prediction
        """
        predictions = []
        
        # Strategy 1: Sequential (if layer_0 -> layer_1)
        if "layer_" in current_key:
            try:
                idx = int(current_key.split("_")[1])
                for i in range(1, count + 1):
                    next_key = f"layer_{idx + i}"
                    if next_key not in predictions:
                        predictions.append(next_key)
            except:
                pass
        
        # Strategy 2: Transition-based
        if self.access_history:
            last = self.access_history[-1] if self.access_history else None
            if last and last != current_key:
                key = (last, current_key)
                if key in self.layer_transitions:
                    # Predict based on transition pattern
                    transitions = {
                        k: v for k, v in self.layer_transitions.items()
                        if k[0] == current_key
                    }
                    if transitions:
                        sorted_trans = sorted(transitions.items(), key=lambda x: -x[1])
                        for k, _ in sorted_trans[:count]:
                            if k[1] not in predictions:
                                predictions.append(k[1])
        
        # Strategy 3: Most frequent (fallback)
        if len(predictions) < count:
            freq: Dict[str, int] = {}
            for key in self.access_history:
                freq[key] = freq.get(key, 0) + 1
            sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
            for key, _ in sorted_freq:
                if key not in predictions and key != current_key:
                    predictions.append(key)
                    if len(predictions) >= count:
                        break
        
        return predictions[:count]
    
    def _record_access(self, key: str, predicted: bool = False) -> None:
        """Record an access for pattern learning."""
        # Update history
        self.access_history.append(key)
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-5000:]
        
        # Update timestamps
        if key not in self.access_timestamps:
            self.access_timestamps[key] = []
        self.access_timestamps[key].append(time.time())
        if len(self.access_timestamps[key]) > 100:
            self.access_timestamps[key] = self.access_timestamps[key][-50:]
        
        # Update transitions
        if len(self.access_history) >= 2:
            prev = self.access_history[-2]
            if prev != key:
                transition = (prev, key)
                self.layer_transitions[transition] = \
                    self.layer_transitions.get(transition, 0) + 1
        
        # Update sequential score
        if len(self.access_history) >= 2:
            prev = self.access_history[-2]
            if self._is_sequential(prev, key):
                self.sequential_score = min(1.0, self.sequential_score + 0.1)
            else:
                self.sequential_score = max(0.0, self.sequential_score - 0.05)
    
    def _is_sequential(self, prev: str, curr: str) -> bool:
        """Check if access appears sequential."""
        if "layer_" in prev and "layer_" in curr:
            try:
                prev_idx = int(prev.split("_")[1])
                curr_idx = int(curr.split("_")[1])
                return curr_idx == prev_idx + 1
            except:
                pass
        return False
    
    def _tier_priority(self, tier: MemoryTier) -> int:
        """Get priority level for a tier (lower = faster)."""
        priorities = {
            MemoryTier.ULTRA_CACHE: 0,
            MemoryTier.HBM: 1,
            MemoryTier.RAM: 2,
            MemoryTier.SSD: 3,
            MemoryTier.GENERATED: 4,
        }
        return priorities.get(tier, 99)
    
    def _promote(self, key: str, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Move data from one tier to another."""
        if key in self.tier_data[from_tier]:
            data = self.tier_data[from_tier][key]
            
            # Remove from old tier
            size = self._estimate_size(data)
            self.stats[from_tier].used_bytes -= size
            
            # Add to new tier
            self.store(key, data, to_tier)
    
    def _auto_evict(self, tier: MemoryTier) -> None:
        """Automatically evict from a tier if full."""
        while (self.stats[tier].used_bytes > self.tier_sizes[tier] and
               self.tier_data[tier]):
            # Evict least recently used
            oldest_key = None
            oldest_time = float('inf')
            
            for key, timestamps in self.access_timestamps.items():
                if key in self.tier_data[tier] and timestamps:
                    if timestamps[-1] < oldest_time:
                        oldest_time = timestamps[-1]
                        oldest_key = key
            
            if oldest_key:
                self._evict(oldest_key, tier)
            else:
                break
    
    def _evict(self, key: str, from_tier: MemoryTier) -> None:
        """Evict a key from a tier."""
        if key in self.tier_data[from_tier]:
            data = self.tier_data[from_tier][key]
            size = self._estimate_size(data)
            
            # Demote to lower tier if possible
            tiers_above = [t for t in MemoryTier 
                         if self._tier_priority(t) > self._tier_priority(from_tier)]
            
            if tiers_above:
                lower_tier = min(tiers_above, key=lambda t: self._tier_priority(t))
                if lower_tier != MemoryTier.GENERATED:
                    self.store(key, data, lower_tier)
            
            # Remove from current tier
            del self.tier_data[from_tier][key]
            self.stats[from_tier].used_bytes -= size
    
    def contains(self, key: str) -> bool:
        """Check if key exists in any tier."""
        return any(key in self.tier_data[tier] for tier in MemoryTier)
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, (int, float, str)):
            return 100  # Rough estimate
        else:
            return 1000  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        summary = {
            "total_tiers": len(MemoryTier),
            "sequential_score": self.sequential_score,
            "unique_keys": len(self.access_timestamps),
            "transition_count": len(self.layer_transitions),
            "tiers": {},
        }
        
        for tier in MemoryTier:
            stats = self.stats[tier]
            summary["tiers"][tier.value] = {
                "size_mb": stats.size_bytes / (1024**2),
                "used_mb": stats.used_bytes / (1024**2),
                "usage_percent": stats.usage_percent,
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "prefetch_hits": stats.prefetch_hits,
            }
        
        return summary
    
    def print_stats(self) -> None:
        """Print memory statistics."""
        logger.info("=" * 60)
        logger.info("COGNITIVE MEMORY STATISTICS")
        logger.info("=" * 60)
        
        for tier in MemoryTier:
            stats = self.stats[tier]
            logger.info(f"{tier.value.upper()}:")
            logger.info(f"  Size: {stats.size_bytes / (1024**2):.0f} MB")
            logger.info(f"  Used: {stats.used_bytes / (1024**2):.0f} MB ({stats.usage_percent:.1f}%)")
            logger.info(f"  Hits: {stats.hits}, Misses: {stats.misses} (Rate: {stats.hit_rate:.2%})")
            logger.info(f"  Prefetch hits: {stats.prefetch_hits}")
        
        logger.info(f"Sequential access score: {self.sequential_score:.2f}")
        logger.info(f"Unique keys tracked: {len(self.access_timestamps)}")
        logger.info("=" * 60)


class UltraFastCache:
    """
    Ultra-fast L1 cache for critical data.
    
    Uses simple array indexing for O(1) access.
    Only stores most recently used items.
    """
    
    def __init__(self, size: int = 64):
        self.size = size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
    
    def put(self, key: str, value: Any) -> None:
        """Add to cache, evicting if necessary."""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.size:
            # Evict oldest
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache
