"""Adaptive-compute probe — measurement harness for Phase 4 of the
post-Transformer research program.

See ../post-transformer-paradigm.md (Phase 4) for the hypothesis this harness is
designed to falsify:

    On a difficulty-heterogeneous request mix, a router that allocates the
    compute budget per request (reasoning length / effective depth) from a cheap
    difficulty estimate matches the quality of a fixed-maximum budget while
    spending strictly fewer FLOPs on average. If the FLOPs saved at iso-quality
    is not significant, the hypothesis is FALSE for this regime.

IMPORTANT — read before citing any number this script prints:

    This file is a *measurement framework*, NOT a result. Out of the box it runs
    against a deterministic synthetic stand-in model (`SyntheticModel`) whose only
    purpose is to exercise the measurement and plotting logic. The synthetic model
    is rigged with a plausible-but-invented relationship between budget and
    quality; running it tells you NOTHING about real LLMs. It exists so the
    accounting code can be reviewed and tested independently of a GPU.

    To produce a real, citable result you must replace `SyntheticModel` with a
    real backend (see `AirLLMBackend` stub) and a real difficulty-heterogeneous
    eval set. Until then, every number here is synthetic by construction.

The accounting (FLOPs-vs-quality Pareto, and whether `adaptive` dominates the
segment between `fixed-low` and `fixed-high`) is the actual contribution. Keep it
honest: the decisive test is the Pareto comparison, not the headline accuracy.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    """One eval item. `difficulty` in [0, 1] is GROUND TRUTH used only to build a
    heterogeneous mix and to sanity-check the estimator — never given to the
    policy at decision time."""

    prompt: str
    answer: str
    difficulty: float


@dataclass
class RunResult:
    policy: str
    accuracy: float
    mean_budget: float
    mean_flops: float  # arbitrary but consistent FLOP units across policies
    n: int

    def __str__(self) -> str:
        return (
            f"{self.policy:>12} | acc={self.accuracy:6.3f} | "
            f"mean_budget={self.mean_budget:6.2f} | "
            f"mean_flops={self.mean_flops:10.1f} (n={self.n})"
        )


# --------------------------------------------------------------------------- #
# Model backend protocol
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    """A backend answers a prompt given an integer compute budget (e.g. number of
    reasoning steps / effective passes) and reports the FLOPs it spent."""

    def run(self, prompt: str, budget: int) -> tuple[str, float]:
        """Return (model_answer, flops_spent)."""
        ...


class SyntheticModel:
    """Deterministic stand-in. DO NOT cite its outputs as evidence about LLMs.

    Encodes one invented assumption purely to exercise the harness: an item of
    difficulty d is answered correctly once the budget exceeds a threshold that
    grows with d, with a little noise. FLOPs are modeled as linear in budget.
    """

    def __init__(self, flops_per_unit: float = 1.0e9, seed: int = 0):
        self.flops_per_unit = flops_per_unit
        self._rng = random.Random(seed)

    def run(self, prompt: str, budget: int) -> tuple[str, float]:
        # Recover the planted difficulty from the example registry (synthetic only).
        d = _SYNTH_DIFFICULTY.get(prompt, 0.5)
        threshold = 1 + d * (MAX_BUDGET - 1)  # easy -> ~1, hard -> ~MAX_BUDGET
        jitter = self._rng.uniform(-0.5, 0.5)
        correct = budget >= (threshold + jitter)
        answer = "CORRECT" if correct else "WRONG"
        flops = budget * self.flops_per_unit
        return answer, flops


class AirLLMBackend:
    """STUB: wire this to AirLLM (see ../../air_llm/) to get real numbers.

    Sketch of the real implementation:
      - load a small/quantized decoder via AirLLM's AutoModel
      - `budget` controls reasoning length (max new tokens for a CoT scaffold) or
        an early-exit depth threshold
      - measure FLOPs from (layers x budget x hidden) or a profiler, consistently
        across policies so the Pareto comparison is apples-to-apples
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "AirLLMBackend is intentionally unimplemented. Replace SyntheticModel "
            "with a real AirLLM-backed model + a real eval set before reporting any "
            "result. See Phase 4 of the research document."
        )

    def run(self, prompt: str, budget: int) -> tuple[str, float]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Difficulty estimator (cheap, must NOT see ground-truth difficulty)
# --------------------------------------------------------------------------- #
def cheap_difficulty_estimate(prompt: str, backend: Backend) -> float:
    """Placeholder for a cheap signal: first-token entropy / margin, prompt length,
    or a tiny classifier. Here we use a length heuristic so the harness runs with
    zero model calls. Replace with a real cheap probe for real experiments.

    Returns an estimate in [0, 1]. Deliberately imperfect: a perfect estimator
    would make the test trivially favorable, which would be dishonest.
    """
    # Length-based proxy, squashed to [0, 1]. Imperfect on purpose.
    return 1.0 - math.exp(-len(prompt) / 40.0)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
MIN_BUDGET = 1
MAX_BUDGET = 16


def policy_fixed(budget: int) -> Callable[[Example, Backend], int]:
    return lambda ex, backend: budget


def policy_adaptive(ex: Example, backend: Backend) -> int:
    """Map a cheap difficulty estimate to a compute budget."""
    est = cheap_difficulty_estimate(ex.prompt, backend)
    return max(MIN_BUDGET, min(MAX_BUDGET, round(MIN_BUDGET + est * (MAX_BUDGET - MIN_BUDGET))))


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate(
    name: str,
    chooser: Callable[[Example, Backend], int],
    data: list[Example],
    backend: Backend,
) -> RunResult:
    n = len(data)
    correct = 0
    total_budget = 0.0
    total_flops = 0.0
    for ex in data:
        budget = chooser(ex, backend)
        ans, flops = backend.run(ex.prompt, budget)
        correct += int(ans == ex.answer)
        total_budget += budget
        total_flops += flops
    return RunResult(
        policy=name,
        accuracy=correct / n,
        mean_budget=total_budget / n,
        mean_flops=total_flops / n,
        n=n,
    )


# --------------------------------------------------------------------------- #
# Synthetic eval set (REPLACE with a real heterogeneous benchmark)
# --------------------------------------------------------------------------- #
_SYNTH_DIFFICULTY: dict[str, float] = {}


def make_synthetic_dataset(n: int = 200, seed: int = 1) -> list[Example]:
    rng = random.Random(seed)
    data: list[Example] = []
    for i in range(n):
        d = rng.random()
        # Harder items get longer prompts so the length-proxy estimator is
        # informative-but-imperfect (correlated, not perfect).
        length = int(5 + d * 80 + rng.uniform(-10, 10))
        prompt = f"item{i}:" + "x" * max(1, length)
        _SYNTH_DIFFICULTY[prompt] = d
        data.append(Example(prompt=prompt, answer="CORRECT", difficulty=d))
    return data


# --------------------------------------------------------------------------- #
# Pareto verdict
# --------------------------------------------------------------------------- #
def verdict(low: RunResult, high: RunResult, adapt: RunResult) -> str:
    """The decisive test: does `adaptive` dominate the FLOPs/quality frontier
    between fixed-low and fixed-high? i.e. match high's accuracy at lower flops,
    or beat low's accuracy without paying high's flops."""
    matches_high_quality = adapt.accuracy >= high.accuracy - 0.01
    cheaper_than_high = adapt.mean_flops < high.mean_flops
    if matches_high_quality and cheaper_than_high:
        saved = 100.0 * (1 - adapt.mean_flops / high.mean_flops)
        return (
            f"ADAPTIVE WINS (synthetic): iso-quality vs fixed-high, "
            f"{saved:.1f}% fewer FLOPs. NOTE: synthetic backend — not evidence about real LLMs."
        )
    return (
        "ADAPTIVE DOES NOT DOMINATE under this configuration. "
        "(With the synthetic backend this is just a property of the toy; "
        "the real test requires AirLLMBackend + a real benchmark.)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="synthetic eval size")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print(__doc__.splitlines()[0])
    print("=" * 72)
    print("WARNING: synthetic backend active. Numbers below are NOT a real result.")
    print("=" * 72)

    data = make_synthetic_dataset(n=args.n, seed=args.seed)
    backend = SyntheticModel(seed=args.seed)

    low = evaluate("fixed-low", policy_fixed(MIN_BUDGET), data, backend)
    high = evaluate("fixed-high", policy_fixed(MAX_BUDGET), data, backend)
    adapt = evaluate("adaptive", policy_adaptive, data, backend)

    for r in (low, high, adapt):
        print(r)
    print("-" * 72)
    print(verdict(low, high, adapt))


if __name__ == "__main__":
    main()
