"""
Memory Manager
============

Integrated memory management for the Neural Runtime OS.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import threading
import time

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory tier levels."""
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"
    GENERATED = "generated"


@dataclass
class MemoryAllocation:
    """A memory allocation."""
    name: str
    size_bytes: int
    tier: MemoryTier
    allocated_at: float
    last_access: float


class MemoryManager:
    """
    Centralized memory management for Neural Runtime.
    
    Features:
    - Multi-tier memory
    - Allocation tracking
    - Memory pressure detection
    - Automatic offloading
    """
    
    def __init__(
        self,
        max_vram_gb: float = 24.0,
        max_ram_gb: float = 128.0,
    ):
        self.max_vram = int(max_vram_gb * 1024**3)
        self.max_ram = int(max_ram_gb * 1024**3)
        
        # Allocations
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.lock = threading.Lock()
        
        # Statistics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.offload_count = 0
    
    def allocate(
        self,
        name: str,
        size_bytes: int,
        tier: MemoryTier = MemoryTier.GPU,
    ) -> bool:
        """Allocate memory."""
        with self.lock:
            # Check capacity
            current_usage = self._get_usage(tier)
            
            if current_usage + size_bytes > self._get_capacity(tier):
                # Try to free space
                if not self._free_space(tier, size_bytes):
                    logger.warning(f"Cannot allocate {size_bytes} bytes in {tier}")
                    return False
            
            # Allocate
            alloc = MemoryAllocation(
                name=name,
                size_bytes=size_bytes,
                tier=tier,
                allocated_at=time.time(),
                last_access=time.time(),
            )
            self.allocations[name] = alloc
            self.allocation_count += 1
            
            return True
    
    def deallocate(self, name: str) -> bool:
        """Deallocate memory."""
        with self.lock:
            if name in self.allocations:
                alloc = self.allocations.pop(name)
                self.deallocation_count += 1
                return True
            return False
    
    def get_tier(self, name: str) -> Optional[MemoryTier]:
        """Get the tier of an allocation."""
        if name in self.allocations:
            return self.allocations[name].tier
        return None
    
    def access(self, name: str) -> None:
        """Record an access to an allocation."""
        if name in self.allocations:
            self.allocations[name].last_access = time.time()
    
    def get_pressure(self) -> Dict[MemoryTier, float]:
        """Get memory pressure for each tier."""
        pressure = {}
        
        for tier in MemoryTier:
            usage = self._get_usage(tier)
            capacity = self._get_capacity(tier)
            pressure[tier.value] = usage / max(capacity, 1)
        
        return pressure
    
    def _get_usage(self, tier: MemoryTier) -> int:
        """Get current usage for a tier."""
        return sum(
            a.size_bytes for a in self.allocations.values()
            if a.tier == tier
        )
    
    def _get_capacity(self, tier: MemoryTier) -> int:
        """Get capacity for a tier."""
        if tier == MemoryTier.GPU:
            return self.max_vram
        elif tier == MemoryTier.CPU:
            return self.max_ram
        else:
            return int(1e12)  # Effectively unlimited
    
    def _free_space(self, tier: MemoryTier, needed_bytes: int) -> bool:
        """Try to free space in a tier."""
        # Find least recently used allocations
        sorted_allocs = sorted(
            self.allocations.items(),
            key=lambda x: x[1].last_access
        )
        
        freed = 0
        to_remove = []
        
        for name, alloc in sorted_allocs:
            if alloc.tier == tier:
                freed += alloc.size_bytes
                to_remove.append(name)
                
                if freed >= needed_bytes:
                    break
        
        # Remove freed allocations
        for name in to_remove:
            self.deallocate(name)
            self.offload_count += 1
        
        return freed >= needed_bytes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_allocations": len(self.allocations),
            "allocation_count": self.allocation_count,
            "deallocation_count": self.deallocation_count,
            "offload_count": self.offload_count,
            "vram_usage_bytes": self._get_usage(MemoryTier.GPU),
            "ram_usage_bytes": self._get_usage(MemoryTier.CPU),
        }
    
    def __repr__(self) -> str:
        return (
            f"MemoryManager(\n"
            f"  allocations: {len(self.allocations)}\n"
            f"  vram: {self._get_usage(MemoryTier.GPU) / 1024**3:.2f} GB\n"
            f"  ram: {self._get_usage(MemoryTier.CPU) / 1024**3:.2f} GB\n"
            f")"
        )


class PooledMemoryManager(MemoryManager):
    """
    Memory manager with memory pooling.
    
    Pools frequently used tensors to avoid reallocation.
    """
    
    def __init__(self, *args, pool_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool: Dict[str, List[torch.Tensor]] = {}
        self.pool_max_size = pool_size
    
    def get_from_pool(
        self,
        name: str,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Get a tensor from the pool."""
        pool_key = f"{name}_{shape}_{dtype}"
        
        if pool_key in self.pool and self.pool[pool_key]:
            tensor = self.pool[pool_key].pop()
            self.access(name)
            return tensor
        
        return None
    
    def return_to_pool(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        """Return a tensor to the pool."""
        pool_key = f"{name}_{tuple(tensor.shape)}_{tensor.dtype}"
        
        if pool_key not in self.pool:
            self.pool[pool_key] = []
        
        if len(self.pool[pool_key]) < self.pool_max_size:
            self.pool[pool_key].append(tensor)
