"""Post-Transformer Runtime — reproducible benchmark harness (Phase 5 baseline).

Public API:
    from post_transformer_runtime.benchmarks import run_benchmark, BenchmarkResult
    from post_transformer_runtime.benchmarks.metrics import HostInfo, human_bytes

The harness is engine-agnostic: any object satisfying the ``EngineAdapter``
protocol (see ``runner.py``) can be measured on the same six axes
(memory / VRAM / storage / throughput / energy / quality). AirLLM is the
reference baseline (``baseline_airllm.py``).
"""

from .metrics import BenchmarkResult, HostInfo, human_bytes  # noqa: F401
from .runner import run_benchmark, EngineAdapter  # noqa: F401

__all__ = [
    "BenchmarkResult",
    "HostInfo",
    "human_bytes",
    "run_benchmark",
    "EngineAdapter",
]
