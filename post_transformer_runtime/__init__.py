"""Post-Transformer Neural Runtime.

Research scaffold exploring whether trillion-parameter-class capability can be
obtained without storing a trillion explicit parameters. See
docs/AIRLLM_AUTOPSY.md (Phase 0) for the scientific framing and guardrails.

Currently implemented: ``benchmarks`` -- the reproducible measurement harness
that fixes the AirLLM baseline every future axis must be compared against.
"""

__all__ = ["benchmarks"]
