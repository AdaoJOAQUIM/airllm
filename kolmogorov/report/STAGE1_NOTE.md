# Stage 1 — Scaling law of compressibility (the decisive gate)

**Date:** 2026-06-28 · **Plan:** `docs/PROJECT_KOLMOGOROV.md` §6 · **Theory:**
`docs/PROOFS_KOLMOGOROV.md` Cor 3 (pincer) · **Pre-registration:**
`kolmogorov/registry/stage1.yaml` · **Reproduce:**
`python3 -m kolmogorov.experiments.stage1_scaling --sizes 70m 160m 410m`

## Question
Does the metric-entropy density `H_ε(N)/n_params` **decrease** with model size?
Operationalized as the distortion incurred by a fixed-rate compressor
(per-channel uniform quantization) as `N` grows. Decreasing ⇒ streaming inference
gets cheaper per parameter with scale (the C1 / RD-COMP win regime). Flat/rising
⇒ disk-streaming is fundamentally `Ω(model)` per token-batch.

## Result (measured, Pythia suite, WikiText-2)

| N | base ppl | Δppl @ 8-bit | Δppl @ 4-bit |
|---:|---:|---:|---:|
| 70M  | 71.50 | +0.668 | +85.1 |
| 160M | 41.45 | +0.291 | +72.5 |
| 410M | 23.76 | +0.094 | +29.8 |
| 1.4B | 18.68 | **+0.016** | **+8.9** |

(1.4B used a slightly shorter eval slice and bit-widths {8,4}; see results JSON.)

**The law is strictly monotone decreasing across a 20× range, at both
bit-widths.** Rough power-law fits `Δppl ∝ N^{-α}`:
- 8-bit: **α ≈ 1.25** (distortion shrinks ~40× over 20× scale),
- 4-bit: **α ≈ 0.75** (distortion shrinks ~10× over 20× scale).

## Interpretation (pre-registered §C1_supported)

The same fixed-rate compression hurts a **larger** model **strictly less**. The
information carried per parameter, at fixed output distortion, **falls with
scale** — exactly the C1-supporting direction, and exactly what the pincer
`S*_ε = Θ(min(n, H_ε))` needs for streaming/generation to win at scale. This is
consistent with the independent quantization literature (larger models quantize
to lower bit-widths more gracefully — GPTQ/AWQ).

**This is the first positive empirical signal for the whole program.** Where
Stage 0 *killed* the naive generator (low-rank), Stage 1 *supports* the core
premise: real weights become more compressible per parameter as models grow.

## What it does and does not establish (honesty)

- **Does:** give strong, monotone, multi-scale evidence that `H_ε(N)/n` is
  decreasing — the sign the theory required.
- **Does not:** prove `H_ε = o(N)` in the limit. Uniform RTN quantization is an
  *upper bound* on `H_ε/n`; four points are a trend, not an asymptote. The
  absolute bits are not yet sub-4-bit because uncalibrated RTN collapses below
  4-bit — a known artifact, not a property of the weights.

## Next
1. **Calibrated quantization (GPTQ-style)** to push the achievable bits down and
   measure `b*(N)` at sub-4-bit, sharpening the density estimate.
2. **More scales** (2.8B, 6.9B) to fit the exponent `α` with confidence intervals
   and test whether `H_ε/n → 0` or plateaus.
3. Feed the measured `H_ε(N)` back into the pincer to predict the achievable
   tokens/s of a generation-based engine vs disk reload (close the loop with the
   roofline simulator, `systems/roofline_sim.py`).

## Decision
Gate **passed in the supporting direction.** C1 remains open (needs the
asymptote), but the program is alive and pointed the right way: theory says the
prize lives where `H_ε ≪ n`, and the data say that regime *opens up with scale*.
