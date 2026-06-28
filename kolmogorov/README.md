# Project KOLMOGOROV

Research code for the hypothesis: **LLM weights can be *generated* on the fly by a
small resident operator instead of being stored and streamed** — turning the I/O
wall of disk-streaming inference (AirLLM's bottleneck) into a compute problem,
where compute is abundant.

Full plan, math, and pre-registered success/refutation criteria:
[`../docs/PROJECT_KOLMOGOROV.md`](../docs/PROJECT_KOLMOGOROV.md).

## Layout
```
systems/roofline_sim.py        # C2 gate: when does recompute beat reload? (numpy only)
generators/lowrank.py          # G1 baseline weight generator (truncated SVD)
eval/perplexity.py             # deterministic perplexity harness
experiments/stage0_lowrank.py  # Stage 0: first b_eff vs Δppl curve on GPT-2
registry/stage0.yaml           # pre-registration manifest (frozen thresholds)
report/                        # results + honest interpretation notes
tests/                         # hermetic unit tests (no network/GPU)
```

## Reproduce
```bash
# C2 systems gate (no downloads):
python3 -m kolmogorov.systems.roofline_sim
python3 -m unittest kolmogorov.tests.test_roofline_sim -v

# Stage 0 C1 probe (downloads GPT-2 + WikiText-2, CPU ok):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets
python3 -m kolmogorov.experiments.stage0_lowrank --max-chunks 30
```

## Status (2026-06-28)
- **C2 (systems):** simulator shows recompute beats reload for any generator of
  rank up to ~6000 on NVMe (≥476 on the tightest realistic tier). C2's fate is
  therefore reduced to C1.
- **C1 (information):** Stage 0 **refutes the naive low-rank generator (G1)** —
  pretrained weights are near-full-rank; a 10% rank cut doubles perplexity. Per
  pre-registration this is a *pivot-generator* signal, not a refutation of C1.
  Next: calibrated/per-layer decomposition, residual-after-quantization, then
  hypernetwork/diffusion generators; and the Stage 1 scaling law.

This is honest, falsifiable, in-progress research. Negative results are reported,
not hidden.
