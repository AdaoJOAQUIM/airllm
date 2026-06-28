# Stage 0 — Results & honest interpretation

**Date:** 2026-06-28 · **Plan:** `docs/PROJECT_KOLMOGOROV.md` · **Pre-registration:**
`kolmogorov/registry/stage0.yaml` · **Reproduce:**
`python3 -m kolmogorov.experiments.stage0_lowrank --max-chunks 30`

## Setup
- Model: GPT-2 small (124M), CPU, seed 0.
- Eval: Salesforce/wikitext (wikitext-2-raw-v1, test), 30 non-overlapping
  512-token chunks, token-level perplexity.
- Generator under test: **G1**, uniform truncated-SVD low-rank on all large 2-D
  block weights (embeddings / lm_head / LayerNorm / biases left intact).
- Generator family G1 is, by design, the **weakest** in the plan — a conservative
  floor, run first because it is the cheapest way to learn the shape of the
  problem.

## Result (measured)

| config | b_eff (bits/w) | perplexity | Δppl |
|--------|---------------:|-----------:|-----:|
| baseline (fp16 nominal)        | 16.0  | 37.30 | 0.00 |
| **control** full-rank SVD      | 16.0  | 37.30 | ~1e-6 |
| rank 0.9                       | 19.19 | 69.47 | +32.2 |
| rank 0.5                       | 10.67 | 3 563 | +3 526 |
| rank 0.25                      | 5.33  | 10 695 | +10 658 |
| rank 0.1                       | 2.14  | 5 841 | +5 804 |
| rank 0.05                      | 1.06  | 6 755 | +6 717 |
| rank 0.02                      | 0.42  | 7 974 | +7 937 |

## What this proves (and what it does not)

1. **The harness is valid.** Full-rank SVD reproduces the baseline to 1e-6. The
   catastrophic numbers below are real, not a bug. (This control is now asserted
   in-code; the experiment fails loudly if it ever breaks.)

2. **G1 is decisively refuted.** Two independent facts kill naive low-rank:
   - Quality collapses immediately: a mere **10 % rank cut nearly doubles
     perplexity** (37 → 69), and errors **compound across the 12 layers** into
     four-digit perplexities.
   - Low-rank does not even *compress* until rank-fraction ≈ 0.5 (factor storage
     `r(d+k)` only beats `d·k` once `r < min/2`). So **there is no operating point
     where G1 both compresses and preserves quality.** The two requirements never
     overlap.

3. **C1 is NOT refuted — this was pre-registered.** Per `stage0.yaml`
   `pivot_signal`: a steep cliff before any compression means *the exploitable
   structure is not plain global low-rank*, which tells us to **change generator
   family, not abandon the hypothesis.** This matches the literature: pretrained
   weight matrices are near-full-rank; it is *fine-tuning updates* (LoRA) and
   *activations* (Deja Vu) that are low-rank — not the raw pretrained weights.

## Scientific value of a negative result

We spent ~no compute and **eliminated the most obvious wrong answer**, sharpening
the real question: *if* trained weights are compressible, the information lives in
a structure that uniform SVD cannot see. That is precisely why the plan reserved
stronger generators. The cheap experiment did its job — it tried to kill the idea
and instead killed the naive version of it.

## Next generators to test (ordered by expected information-per-cost)
1. **Per-layer rank allocation + calibration** — activation/Hessian-weighted
   decomposition (à la SVD-LLM / ASVD), non-uniform ranks; some layers tolerate
   far more reduction than others.
2. **Compress the residual after quantization**, not the raw fp16 weights — the
   C1✓/C2-friendly pivot that survives even if recompute never beats reload.
3. **Hypernetwork / coordinate-MLP (G2)** — capture cross-layer shared structure
   a per-matrix SVD cannot.
4. **Diffusion over weights (G3)** — upper bound on reconstruction quality.

## Decision
Gate **not** passed for G1. Hypothesis C1 remains open. Proceed to G2/calibrated
decomposition at Stage 0/1 before any large-model spend (plan §11.3). Recommend
also adding Stage 1 scaling (Pythia-160M → 1.4B) to measure whether b_eff at fixed
Δppl improves with size — the decisive scaling law.
