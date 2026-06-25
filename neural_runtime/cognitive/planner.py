"""
Cognitive Planner
============

A planner that analyzes queries and creates optimal inference plans.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Type of inference task."""
    SIMPLE_QA = "simple_qa"           # Simple factual question
    REASONING = "reasoning"           # Multi-step reasoning
    CODE_GENERATION = "code"          # Code generation
    CREATIVE = "creative"             # Creative writing
    LONG_CONTEXT = "long_context"     # Long context processing
    DOMAIN_SPECIFIC = "domain"        # Domain-specific knowledge


class PlanStage(Enum):
    """Stage of an inference plan."""
    PARSE = "parse"
    RETRIEVE = "retrieve"
    REASON = "reason"
    GENERATE = "generate"
    REFINE = "refine"


@dataclass
class InferencePlan:
    """An optimized plan for inference."""
    task_type: TaskType
    complexity_score: float
    stages: List[PlanStage]
    estimated_layers_needed: int
    estimated_memory_mb: int
    strategy: str
    optimizations: List[str]


class CognitivePlanner:
    """
    Cognitive Planner for query analysis and plan optimization.
    
    Analyzes queries to determine:
    1. Task type and complexity
    2. Required reasoning depth
    3. Knowledge retrieval needs
    4. Optimal inference strategy
    5. Resource allocation
    
    ┌────────────────────────────────────────────────────────────────┐
    │                    COGNITIVE PLANNER FLOW                        │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   Query ──► Analyzer ──► Planner ──► Executor              │
    │                  │              │              │               │
    │                  ▼              ▼              ▼               │
    │             Task Type      Optimal       Inference            │
    │             Complexity      Plan          with Plan           │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self):
        # Keywords for task classification
        self.task_keywords = {
            TaskType.SIMPLE_QA: [
                "what", "who", "when", "where", "which", "define", "name"
            ],
            TaskType.REASONING: [
                "why", "how", "explain", "prove", "deduce", "calculate",
                "analyze", "compare", "contrast", "if", "then", "because"
            ],
            TaskType.CODE_GENERATION: [
                "code", "function", "python", "javascript", "java",
                "write", "implement", "algorithm", "class", "def"
            ],
            TaskType.CREATIVE: [
                "story", "write", "poem", "creative", "imagine",
                "describe", "invent", "dream", "fantasy"
            ],
            TaskType.LONG_CONTEXT: [
                "document", "paper", "article", "chapter", "paragraph",
                "summary", "context", "given", "above", "following"
            ],
        }
        
        # Layer estimates per task type
        self.layer_estimates = {
            TaskType.SIMPLE_QA: 20,          # Fewer layers needed
            TaskType.REASONING: 80,           # Full depth
            TaskType.CODE_GENERATION: 60,     # Medium-full
            TaskType.CREATIVE: 40,           # Medium
            TaskType.LONG_CONTEXT: 80,        # Full
            TaskType.DOMAIN_SPECIFIC: 50,    # Medium-full
        }
    
    def analyze(self, query: str, context: Optional[str] = None) -> InferencePlan:
        """
        Analyze a query and create an inference plan.
        
        Args:
            query: The user's query
            context: Optional context (previous conversation, documents, etc.)
            
        Returns:
            An optimized inference plan
        """
        # Classify task type
        task_type = self._classify_task(query)
        
        # Estimate complexity
        complexity = self._estimate_complexity(query, context)
        
        # Determine required layers
        layers_needed = self._estimate_layers(task_type, complexity)
        
        # Create execution plan
        stages = self._create_stages(task_type, complexity)
        
        # Determine optimizations
        optimizations = self._select_optimizations(task_type, complexity)
        
        # Estimate memory
        memory_mb = self._estimate_memory(task_type, layers_needed)
        
        # Select overall strategy
        strategy = self._select_strategy(task_type, complexity)
        
        return InferencePlan(
            task_type=task_type,
            complexity_score=complexity,
            stages=stages,
            estimated_layers_needed=layers_needed,
            estimated_memory_mb=memory_mb,
            strategy=strategy,
            optimizations=optimizations,
        )
    
    def _classify_task(self, query: str) -> TaskType:
        """Classify the task type based on keywords."""
        query_lower = query.lower()
        scores = {}
        
        for task_type, keywords in self.task_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[task_type] = score
        
        if not scores or max(scores.values()) == 0:
            return TaskType.SIMPLE_QA
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _estimate_complexity(self, query: str, context: Optional[str]) -> float:
        """
        Estimate query complexity from 0.0 to 1.0.
        
        Factors:
        - Query length
        - Number of entities
        - Reasoning requirements
        - Context length
        """
        complexity = 0.0
        
        # Length factor
        words = query.split()
        complexity += min(len(words) / 50, 0.3)
        
        # Reasoning indicators
        reasoning_words = ["because", "therefore", "however", "although",
                         "thus", "hence", "consequently"]
        complexity += sum(0.05 for w in reasoning_words if w in query.lower())
        
        # Multi-part questions
        if " and " in query or " or " in query:
            complexity += 0.1
        
        # Context factor
        if context:
            complexity += min(len(context) / 10000, 0.2)
        
        return min(complexity, 1.0)
    
    def _estimate_layers(self, task_type: TaskType, complexity: float) -> int:
        """Estimate required layers based on task and complexity."""
        base_layers = self.layer_estimates.get(task_type, 40)
        
        # Adjust for complexity
        adjusted = int(base_layers * (0.5 + 0.5 * complexity))
        
        return min(max(adjusted, 1), 80)  # Clamp to 1-80
    
    def _create_stages(self, task_type: TaskType, complexity: float) -> List[PlanStage]:
        """Create execution stages for the task."""
        stages = [PlanStage.PARSE]
        
        if complexity > 0.3:
            stages.append(PlanStage.RETRIEVE)
        
        stages.append(PlanStage.REASON)
        
        if complexity > 0.6:
            stages.append(PlanStage.REFINE)
        
        stages.append(PlanStage.GENERATE)
        
        return stages
    
    def _select_optimizations(self, task_type: TaskType, complexity: float) -> List[str]:
        """Select appropriate optimizations."""
        opts = []
        
        if task_type == TaskType.SIMPLE_QA:
            opts.extend(["early_exit", "kv_cache"])
        elif task_type == TaskType.REASONING:
            opts.extend(["full_depth", "chain_of_thought"])
        elif task_type == TaskType.LONG_CONTEXT:
            opts.extend(["ring_attention", "sparse_attention"])
        elif task_type == TaskType.CODE_GENERATION:
            opts.extend(["syntax_aware", "completion"])
        
        if complexity < 0.3:
            opts.append("aggressive_pruning")
        elif complexity > 0.7:
            opts.append("high_precision")
        
        return opts
    
    def _estimate_memory(self, task_type: TaskType, layers: int) -> int:
        """Estimate memory requirements in MB."""
        # Rough estimate: ~100MB per layer for attention
        return layers * 100
    
    def _select_strategy(self, task_type: TaskType, complexity: float) -> str:
        """Select overall inference strategy."""
        if task_type == TaskType.LONG_CONTEXT:
            return "streaming"
        elif complexity < 0.3:
            return "fast_partial"
        elif complexity > 0.7:
            return "thorough"
        else:
            return "balanced"
    
    def explain_plan(self, plan: InferencePlan) -> str:
        """Generate human-readable explanation of the plan."""
        return f"""
Inference Plan
==============
Task Type: {plan.task_type.value}
Complexity: {plan.complexity_score:.0%}
Layers Needed: {plan.estimated_layers_needed}
Est. Memory: {plan.estimated_memory_mb} MB
Strategy: {plan.strategy}

Stages: {' → '.join(s.value for s in plan.stages)}

Optimizations: {', '.join(plan.optimizations)}
"""
