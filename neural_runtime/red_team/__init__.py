"""
RED TEAM MODULE
==============

MISSION: Destroy the project's hypotheses.

This is NOT part of the project team.
This is the independent adversary whose job is to:
1. Find counter-examples
2. Expose flaws
3. Prove limits
4. Refute claims

Every hypothesis must survive this team to be accepted.
"""

from .adversary import HypothesisAdversary, CounterExample
from .critic import TheoreticalCritic
from .breaker import SystemBreaker

__all__ = ["HypothesisAdversary", "CounterExample", "TheoreticalCritic", "SystemBreaker"]
