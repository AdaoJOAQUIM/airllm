"""
Tests for Parameter Virtualization Engine
====================================

Unit tests for the ParameterVirtualizer and related classes.
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import the module
import sys
sys.path.insert(0, '.')

from neural_runtime.parameter_virt.virtualizer import (
    ParameterVirtualizer,
    ParameterState,
    VirtualizationStrategy,
    VirtualParameter,
    VirtualizationStats,
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.bias1 = nn.Parameter(torch.zeros(256))
        self.bias2 = nn.Parameter(torch.zeros(128))


class TestParameterState:
    """Tests for ParameterState enum."""
    
    def test_all_states_defined(self):
        """Test all parameter states are defined."""
        assert ParameterState.ACTIVE.value == "active"
        assert ParameterState.DORMANT.value == "dormant"
        assert ParameterState.COMPRESSED.value == "compressed"
        assert ParameterState.GENERATED.value == "generated"
        assert ParameterState.APPROXIMATED.value == "approximated"
    
    def test_state_count(self):
        """Test we have 5 states."""
        assert len(ParameterState) == 5


class TestVirtualizationStrategy:
    """Tests for VirtualizationStrategy enum."""
    
    def test_all_strategies_defined(self):
        """Test all strategies are defined."""
        assert VirtualizationStrategy.SPARSE.value == "sparse"
        assert VirtualizationStrategy.MOE.value == "moe"
        assert VirtualizationStrategy.HYPERNETWORK.value == "hypernetwork"
        assert VirtualizationStrategy.COMPRESSED.value == "compressed"
        assert VirtualizationStrategy.TIERED.value == "tiered"
        assert VirtualizationStrategy.ADAPTIVE.value == "adaptive"
    
    def test_strategy_count(self):
        """Test we have 6 strategies."""
        assert len(VirtualizationStrategy) == 6


class TestVirtualParameter:
    """Tests for VirtualParameter dataclass."""
    
    def test_creation(self):
        """Test creating a virtual parameter."""
        vp = VirtualParameter(
            name="layer.weight",
            shape=(512, 256),
            logical_size_bytes=524288,
            actual_size_bytes=524288,
            state=ParameterState.ACTIVE,
        )
        
        assert vp.name == "layer.weight"
        assert vp.shape == (512, 256)
        assert vp.state == ParameterState.ACTIVE
        assert vp.access_count == 0
        assert vp.importance_score == 0.0
    
    def test_default_state(self):
        """Test default state is ACTIVE."""
        vp = VirtualParameter(
            name="test",
            shape=(10, 10),
            logical_size_bytes=400,
            actual_size_bytes=400,
        )
        
        assert vp.state == ParameterState.ACTIVE
    
    def test_access_tracking(self):
        """Test access count increment."""
        vp = VirtualParameter(
            name="test",
            shape=(10, 10),
            logical_size_bytes=400,
            actual_size_bytes=400,
        )
        
        initial_access = vp.access_count
        vp.access_count += 1
        
        assert vp.access_count == initial_access + 1


class TestVirtualizationStats:
    """Tests for VirtualizationStats dataclass."""
    
    def test_creation(self):
        """Test creating stats."""
        stats = VirtualizationStats()
        
        assert stats.total_parameters == 0
        assert stats.active_parameters == 0
        assert stats.dormant_parameters == 0
        assert stats.compression_ratio == 0.0
    
    def test_with_values(self):
        """Test stats with values."""
        stats = VirtualizationStats(
            total_parameters=100,
            active_parameters=10,
            dormant_parameters=90,
            compression_ratio=10.0,
            logical_memory_gb=70.0,
            actual_memory_gb=7.0,
        )
        
        assert stats.total_parameters == 100
        assert stats.active_parameters == 10
        assert stats.dormant_parameters == 90
        assert stats.compression_ratio == 10.0
        assert stats.logical_memory_gb == 70.0
        assert stats.actual_memory_gb == 7.0


class TestParameterVirtualizer:
    """Tests for ParameterVirtualizer class."""
    
    def test_initialization(self):
        """Test virtualizer initialization."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(
            model=model,
            strategy=VirtualizationStrategy.ADAPTIVE,
            target_active_ratio=0.1,
            device="cpu",  # Use CPU since CUDA not available
        )
        
        assert virtualizer.strategy == VirtualizationStrategy.ADAPTIVE
        assert virtualizer.target_active_ratio == 0.1
        assert len(virtualizer.parameters) > 0
    
    def test_parameter_registration(self):
        """Test all model parameters are registered."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        # Should have 8 parameters (3 weights + 3 biases in linear layers + 2 standalone biases)
        assert len(virtualizer.parameters) >= 6
        
        # Check all expected parameters are registered
        param_names = list(virtualizer.parameters.keys())
        assert "linear1.weight" in param_names
        assert "linear1.bias" in param_names
        assert "linear2.weight" in param_names
    
    def test_parameter_states(self):
        """Test all parameters start as ACTIVE."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        for name, vp in virtualizer.parameters.items():
            assert vp.state == ParameterState.ACTIVE
    
    def test_get_parameter_active(self):
        """Test getting an active parameter."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        # Get should work for active parameters
        param = virtualizer.get_parameter("linear1.weight")
        assert param is not None
        # Linear layer: out_features, in_features - for Linear(512, 256) it's [256, 512]
        assert param.shape == (256, 512)
    
    def test_get_nonexistent_parameter(self):
        """Test getting a non-existent parameter returns None."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        param = virtualizer.get_parameter("nonexistent.weight")
        assert param is None
    
    def test_analyze_importance(self):
        """Test importance analysis."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        # Run analysis
        virtualizer.analyze_importance()
        
        # All parameters should have an importance score
        for name, vp in virtualizer.parameters.items():
            assert vp.importance_score >= 0.0
    
    def test_virtualize(self):
        """Test virtualization."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        # Virtualize to 50% active
        virtualizer.virtualize(target_active_ratio=0.5)
        
        stats = virtualizer.get_stats()
        
        # Should have some dormant or compressed parameters (not all active)
        assert stats.active_parameters < stats.total_parameters
    
    def test_compress_parameter(self):
        """Test parameter compression."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        # Compress a parameter
        result = virtualizer.compress_parameter("linear1.weight")
        
        # Should succeed
        assert result == True
        
        # Check the parameter state changed
        assert virtualizer.parameters["linear1.weight"].state == ParameterState.COMPRESSED
    
    def test_compress_nonexistent(self):
        """Test compressing non-existent parameter fails."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        result = virtualizer.compress_parameter("nonexistent")
        assert result == False
    
    def test_activate_parameter(self):
        """Test activating a dormant parameter."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        # First compress
        virtualizer.compress_parameter("linear1.weight")
        assert virtualizer.parameters["linear1.weight"].state == ParameterState.COMPRESSED
        
        # Then activate
        result = virtualizer.activate_parameter("linear1.weight")
        assert result == True
        
        # Should be active now
        assert virtualizer.parameters["linear1.weight"].state == ParameterState.ACTIVE
    
    def test_get_stats(self):
        """Test getting statistics."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        stats = virtualizer.get_stats()
        
        assert stats.total_parameters >= 6
        assert stats.logical_memory_gb > 0
    
    def test_summary(self):
        """Test summary generation."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        summary = virtualizer.summary()
        
        assert isinstance(summary, str)
        assert "Parameter Virtualization Summary" in summary
        assert "Total Parameters" in summary
    
    def test_access_tracking(self):
        """Test access tracking works."""
        model = DummyModel()
        
        virtualizer = ParameterVirtualizer(model=model, device="cpu")
        
        initial_count = virtualizer.parameters["linear1.weight"].access_count
        
        # Access the parameter
        virtualizer.get_parameter("linear1.weight")
        
        assert virtualizer.parameters["linear1.weight"].access_count > initial_count


class TestAdaptiveParameterVirtualizer:
    """Tests for AdaptiveParameterVirtualizer."""
    
    def test_initialization(self):
        """Test adaptive virtualizer initialization."""
        from neural_runtime.parameter_virt.virtualizer import AdaptiveParameterVirtualizer
        
        model = DummyModel()
        
        virtualizer = AdaptiveParameterVirtualizer(
            model=model,
            adaptation_interval=50,
        )
        
        assert virtualizer.adaptation_interval == 50
        assert len(virtualizer.access_counts) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
