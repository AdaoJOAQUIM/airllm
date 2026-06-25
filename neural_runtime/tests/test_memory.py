"""
Tests for Hierarchical Memory System
"""

import pytest
import torch
from neural_runtime.memory import HierarchicalMemory, MemoryLevel, CachePolicy


class TestHierarchicalMemory:
    """Test cases for HierarchicalMemory."""
    
    def test_initialization(self):
        """Test memory system initialization."""
        memory = HierarchicalMemory(
            max_vram_gb=8.0,
            max_ram_gb=32.0,
        )
        
        assert memory.max_vram_bytes == int(8.0 * 1024**3)
        assert memory.max_ram_bytes == int(32.0 * 1024**3)
        assert memory.cache_policy == CachePolicy.ADAPTIVE
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving tensors."""
        memory = HierarchicalMemory(max_vram_gb=8.0)
        
        # Create a test tensor (1MB)
        tensor = torch.randn(128, 1024)  # ~1MB
        
        # Store
        success = memory.store("layer_0", tensor, MemoryLevel.HBM)
        assert success
        
        # Retrieve
        retrieved = memory.get("layer_0")
        assert retrieved is not None
        assert torch.allclose(tensor, retrieved)
    
    def test_eviction(self):
        """Test LRU eviction."""
        # Small capacity to trigger eviction
        memory = HierarchicalMemory(
            max_vram_gb=0.001,  # Very small
            max_ram_gb=1.0,
            cache_policy=CachePolicy.LRU,
        )
        
        # Store multiple layers
        for i in range(10):
            tensor = torch.randn(1000, 1000)  # ~8MB each
            memory.store(f"layer_{i}", tensor, MemoryLevel.HBM)
        
        # First layers should be evicted
        # (depending on exact sizes)
    
    def test_promotion(self):
        """Test layer promotion between tiers."""
        memory = HierarchicalMemory(max_vram_gb=8.0, max_ram_gb=16.0)
        
        # Store in RAM
        tensor = torch.randn(1000, 1000)
        memory.store("layer_0", tensor, MemoryLevel.RAM)
        
        # Retrieve with target HBM - should promote
        retrieved = memory.get("layer_0", target_level=MemoryLevel.HBM)
        assert retrieved is not None
    
    def test_prefetch(self):
        """Test async prefetching."""
        memory = HierarchicalMemory(max_vram_gb=8.0)
        
        # Start prefetch
        futures = memory.prefetch(["layer_0", "layer_1", "layer_2"])
        
        # Wait for results
        for name, future in futures.items():
            result = future.result(timeout=5.0)
    
    def test_prediction(self):
        """Test layer prediction."""
        memory = HierarchicalMemory(max_vram_gb=8.0)
        
        # Record some accesses
        for _ in range(5):
            tensor = torch.randn(100, 100)
            memory.store("layer_0", tensor, MemoryLevel.HBM)
            memory.get("layer_0")
        
        # Predict next layers
        predictions = memory.predict_next_layers(2, count=3)
        assert len(predictions) <= 3
    
    def test_stats(self):
        """Test memory statistics."""
        memory = HierarchicalMemory(max_vram_gb=8.0)
        
        # Use memory
        tensor = torch.randn(1000, 1000)
        memory.store("layer_0", tensor, MemoryLevel.HBM)
        memory.get("layer_0")
        
        # Get stats
        stats = memory.get_stats()
        assert MemoryLevel.HBM in stats
        assert stats[MemoryLevel.HBM].hits >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
