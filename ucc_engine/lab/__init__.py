"""
Scientific Laboratory for UCCE
============================

Every hypothesis must pass through this lab before integration.

Hypothesis → Simulation → Refutation → Validation → Jury Decision
"""

from .hypothesis import Hypothesis, EvidenceLevel, HypothesisLab, Experiment
from .refuter import RefutationAgent, CounterExample, Refutation, RefutationStrength
from .validator import Validator, ValidationResult, Metric, MetricType
from .jury import Jury, JuryDecision, JuryVerdict, JuryScore, JuryMember

__all__ = [
    # Hypothesis
    "Hypothesis",
    "EvidenceLevel",
    "Experiment",
    "HypothesisLab",
    # Refuter
    "RefutationAgent",
    "CounterExample",
    "Refutation",
    "RefutationStrength",
    # Validator
    "Validator",
    "ValidationResult",
    "Metric",
    "MetricType",
    # Jury
    "Jury",
    "JuryDecision",
    "JuryVerdict",
    "JuryScore",
    "JuryMember",
]
