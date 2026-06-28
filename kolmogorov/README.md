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
- **Theory:** `docs/THEOREM_KOLMOGOROV.md` + `docs/PROOFS_KOLMOGOROV.md` — proven
  streaming lower bounds (Thm 1 worst-case, Thm 2 distortion-aware) giving the
  pincer `S*_ε = Θ(min(n, H_ε))`; the RD-COMP conjecture isolated to one open
  lemma (no faked proof).
- **C2 (systems):** simulator shows recompute beats reload for any generator of
  rank up to ~6000 on NVMe (≥476 on the tightest realistic tier). C2's fate is
  therefore reduced to C1.
- **C1 (information):**
  - **Stage 0** *refutes* the naive low-rank generator (weights are near-full-
    rank) — a pre-registered *pivot-generator* signal, not a refutation of C1.
  - **Stage 1** *supports* C1: across Pythia 70M→1.4B, fixed-rate quantization
    distortion falls monotonically (`Δppl ∝ N^{-1.25}` at 8-bit) — the
    metric-entropy density `H_ε/n` decreases with scale, the direction the
    pincer needs. Asymptote still open (needs calibrated quant + larger scales).

This is honest, falsifiable, in-progress research. Negative results (Stage 0) and
positive ones (Stage 1) are reported alike.
