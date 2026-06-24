"""
Universal Cognitive Compression Engine (UCCE)
==========================================

Research platform for AI compression.

Mission:
    Transform AI from weight files to generative structures.
    Explore whether multi-terabyte models can be represented
    as much smaller generative structures.

Key Principles:
1. Every hypothesis must pass scientific validation
2. Every idea must survive refutation attempts
3. Every method must be measured objectively
4. Truth > Ambition

Modules:
    lab/         - Scientific laboratory
    model_analyzer/ - Model analysis (requires torch)
    compression/  - Established compression methods
    cognitive/   - Experimental representations
    runtime/     - Adaptive execution
    format/      - .cog format
"""

__version__ = "0.1.0"
__author__ = "UCC Engine Research Team"

# Lab module (no external dependencies)
from .lab import (
    Hypothesis,
    EvidenceLevel,
    HypothesisLab,
    Experiment,
    RefutationAgent,
    CounterExample,
    Refutation,
    RefutationStrength,
    Validator,
    ValidationResult,
    Metric,
    MetricType,
    Jury,
    JuryDecision,
    JuryVerdict,
    JuryScore,
    JuryMember,
)

__all__ = [
    # Lab
    "Hypothesis",
    "EvidenceLevel",
    "HypothesisLab",
    "Experiment",
    "RefutationAgent",
    "CounterExample",
    "Refutation",
    "RefutationStrength",
    "Validator",
    "ValidationResult",
    "Metric",
    "MetricType",
    "Jury",
    "JuryDecision",
    "JuryVerdict",
    "JuryScore",
    "JuryMember",
]


# Optional: Model Analyzer (requires torch)
def get_model_analyzer():
    """Get ModelAnalyzer if torch is available."""
    try:
        from .model_analyzer import ModelAnalyzer, AnalysisResult
        return ModelAnalyzer, AnalysisResult
    except ImportError:
        return None, None
