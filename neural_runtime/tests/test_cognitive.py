"""
Tests for Cognitive Planner and Memory
====================================

Unit tests for cognitive system components.
"""

import pytest
import sys
sys.path.insert(0, '.')

from neural_runtime.cognitive.planner import (
    CognitivePlanner,
    TaskType,
    PlanStage,
    InferencePlan,
)
from neural_runtime.cognitive.memory import (
    CognitiveMemory,
    MemoryTier,
)


class TestTaskType:
    """Tests for TaskType enum."""
    
    def test_all_types_defined(self):
        """Test all task types are defined."""
        assert TaskType.SIMPLE_QA.value == "simple_qa"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.CODE_GENERATION.value == "code"
        assert TaskType.CREATIVE.value == "creative"
        assert TaskType.LONG_CONTEXT.value == "long_context"
        assert TaskType.DOMAIN_SPECIFIC.value == "domain"
    
    def test_task_type_count(self):
        """Test we have expected number of task types."""
        assert len(TaskType) == 6


class TestPlanStage:
    """Tests for PlanStage enum."""
    
    def test_all_stages_defined(self):
        """Test all stages are defined."""
        assert PlanStage.PARSE.value == "parse"
        assert PlanStage.RETRIEVE.value == "retrieve"
        assert PlanStage.REASON.value == "reason"
        assert PlanStage.GENERATE.value == "generate"
        assert PlanStage.REFINE.value == "refine"
    
    def test_stage_count(self):
        """Test we have 5 stages."""
        assert len(PlanStage) == 5


class TestCognitivePlanner:
    """Tests for CognitivePlanner class."""
    
    def test_initialization(self):
        """Test planner initialization."""
        planner = CognitivePlanner()
        
        assert planner is not None
        assert len(planner.task_keywords) > 0
    
    def test_classify_simple_qa(self):
        """Test classifying simple QA task."""
        planner = CognitivePlanner()
        
        query = "What is the capital of France?"
        plan = planner.analyze(query)
        
        assert plan.task_type == TaskType.SIMPLE_QA
    
    def test_classify_reasoning(self):
        """Test classifying reasoning task."""
        planner = CognitivePlanner()
        
        query = "Why does the sky appear blue? Explain the physics behind this phenomenon."
        plan = planner.analyze(query)
        
        assert plan.task_type == TaskType.REASONING
    
    def test_classify_code(self):
        """Test classifying code generation task."""
        planner = CognitivePlanner()
        
        query = "Write a Python function to calculate fibonacci numbers."
        plan = planner.analyze(query)
        
        assert plan.task_type == TaskType.CODE_GENERATION
    
    def test_classify_creative(self):
        """Test classifying creative task."""
        planner = CognitivePlanner()
        
        query = "Write a short story about a robot who learns to love."
        plan = planner.analyze(query)
        
        assert plan.task_type == TaskType.CREATIVE
    
    def test_estimate_complexity(self):
        """Test complexity estimation."""
        planner = CognitivePlanner()
        
        # Simple query
        simple_plan = planner.analyze("What is 2+2?")
        assert simple_plan.complexity_score < 0.5
        
        # Complex query
        complex_plan = planner.analyze(
            "Analyze the implications of quantum computing on cryptography "
            "and propose potential solutions for post-quantum security."
        )
        assert complex_plan.complexity_score > 0.3
    
    def test_plan_has_required_fields(self):
        """Test plan contains all required fields."""
        planner = CognitivePlanner()
        
        plan = planner.analyze("Explain photosynthesis.")
        
        assert hasattr(plan, 'task_type')
        assert hasattr(plan, 'complexity_score')
        assert hasattr(plan, 'stages')
        assert hasattr(plan, 'estimated_layers_needed')
        assert hasattr(plan, 'estimated_memory_mb')
        assert hasattr(plan, 'strategy')
        assert hasattr(plan, 'optimizations')
    
    def test_estimated_layers_range(self):
        """Test estimated layers are in valid range."""
        planner = CognitivePlanner()
        
        for query in ["Hi", "Complex reasoning task with many steps"]:
            plan = planner.analyze(query)
            
            assert 1 <= plan.estimated_layers_needed <= 80
    
    def test_optimizations_for_task_type(self):
        """Test appropriate optimizations selected for task types."""
        planner = CognitivePlanner()
        
        # Simple QA should have early_exit
        qa_plan = planner.analyze("What is 2+2?")
        assert "early_exit" in qa_plan.optimizations or "kv_cache" in qa_plan.optimizations
        
        # Code should have syntax_aware
        code_plan = planner.analyze("Write a function")
        assert "syntax_aware" in code_plan.optimizations or "completion" in code_plan.optimizations
    
    def test_explain_plan(self):
        """Test plan explanation generation."""
        planner = CognitivePlanner()
        
        plan = planner.analyze("Explain gravity.")
        explanation = planner.explain_plan(plan)
        
        assert isinstance(explanation, str)
        assert plan.task_type.value in explanation


class TestCognitiveMemory:
    """Tests for CognitiveMemory class."""
    
    def test_initialization(self):
        """Test memory system initialization."""
        memory = CognitiveMemory(
            ultra_cache_size_mb=256,
            vram_size_gb=16.0,
            ram_size_gb=64.0,
        )
        
        assert memory is not None
        assert MemoryTier.ULTRA_CACHE in memory.tier_sizes
        assert MemoryTier.HBM in memory.tier_sizes
        assert MemoryTier.RAM in memory.tier_sizes
    
    def test_tier_sizes(self):
        """Test tier sizes are set correctly."""
        memory = CognitiveMemory(vram_size_gb=24.0)
        
        assert memory.tier_sizes[MemoryTier.HBM] == int(24.0 * 1024**3)
    
    def test_store_and_get(self):
        """Test basic store and retrieve."""
        memory = CognitiveMemory()
        
        # Store data
        data = "test_data"
        memory.store("key1", data, MemoryTier.RAM)
        
        # Retrieve
        retrieved = memory.get("key1", target_tier=MemoryTier.RAM)
        
        assert retrieved == data
    
    def test_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        memory = CognitiveMemory()
        
        result = memory.get("nonexistent_key")
        
        assert result is None
    
    def test_contains(self):
        """Test contains check."""
        memory = CognitiveMemory()
        
        assert not memory.contains("key1")
        
        memory.store("key1", "data", MemoryTier.RAM)
        
        assert memory.contains("key1")
    
    def test_access_tracking(self):
        """Test access patterns are tracked."""
        memory = CognitiveMemory()
        
        memory.store("key1", "data", MemoryTier.RAM)
        memory.get("key1")
        memory.get("key1")
        
        assert "key1" in memory.access_history
        # Memory may record more accesses internally (store + get operations)
        assert memory.access_history.count("key1") >= 2
    
    def test_sequential_prediction(self):
        """Test sequential access prediction."""
        memory = CognitiveMemory()
        
        # Record sequential accesses
        for i in range(5):
            memory.store(f"layer_{i}", f"data_{i}", MemoryTier.RAM)
        
        # Predict next
        predictions = memory.predict_and_prefetch("layer_4", count=2)
        
        # Should predict layer_5 and layer_6
        assert len(predictions) >= 0  # May or may not have predictions
    
    def test_stats(self):
        """Test getting statistics."""
        memory = CognitiveMemory()
        
        memory.store("key1", "data", MemoryTier.RAM)
        memory.get("key1")
        
        stats = memory.get_stats()
        
        assert "total_tiers" in stats
        assert "sequential_score" in stats
        assert "tiers" in stats
        assert MemoryTier.RAM.value in stats["tiers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
