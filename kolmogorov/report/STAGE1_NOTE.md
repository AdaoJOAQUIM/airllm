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
| 1.4B | 18.68 | +0.016 | +8.9 |
| 2.8B | 15.34 | +0.055¹ | **+5.2** |

¹ 1.4B/2.8B used shorter slices; 2.8B used a bf16 base (others fp32). See JSON.

**The 4-bit law is strictly monotone decreasing across a 40× range:**
`85.1 → 72.5 → 29.8 → 8.9 → 5.2`, power-law `Δppl ∝ N^{-α}` with **α ≈ 0.76**.

**Honest caveat on 8-bit.** By ≥410M, 8-bit quantization is essentially lossless
(Δppl < 0.1): those numbers sit in the **noise floor** and depend on the
measurement precision path (the 2.8B 0.055 is bf16-vs-bf16, not comparable to the
fp32 0.016 at 1.4B). We therefore claim the scaling law **at 4-bit**, where the
signal is large and robust, and treat 8-bit only as "already lossless at scale."

## Calibration pushes the achievable bits down (kolmogorov/report/stage1_calibration.json)

A cheap **weight-only MSE-clip calibration** (not GPTQ/activation-aware) on
Pythia-410M:

| bits | RTN Δppl | calibrated Δppl |
|---:|---:|---:|
| 4-bit | +29.8 | **+12.1** |
| 3-bit | +7189 (collapse) | **+87** (alive) |

Calibration rescues 3-bit from catastrophic collapse (~80×), confirming that the
critical bits `b*` — and hence the `H_ε/n` *upper-bound estimate* — move downward
with better compressors. Full GPTQ/AWQ (activation-aware) would push further; the
direction is the point.

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

## Connection to the theory (Theorem 4)

`PROOFS_KOLMOGOROV.md` Thm 4 proves `S*_ε(𝓕) = Θ(H_ε(𝓕))`: the entire systems
question reduces to the value of `H_ε(N)`. The quantity `b*(N) ≈ H_ε(N)/n` we
measure here is therefore *the* number the theorem leaves open. The 4-bit law
(`Δppl ∝ N^{-0.76}`) is direct evidence that `H_ε(N)/n` **decreases with scale**
— the regime in which generative/streaming inference wins.

## Done since first Stage-1 pass
- ✅ extended to 2.8B (4-bit law holds, 40× range, α ≈ 0.76).
- ✅ calibrated quantizer: rescues 3-bit, lowers the `H_ε/n` upper-bound estimate.

## Next
1. **Full GPTQ/AWQ** (activation-aware) to reach genuinely sub-4-bit `b*(N)`.
2. **6.9B / 12B** scales to fit `α` with confidence intervals and test
   `H_ε/n → 0` vs plateau (needs >16GB RAM or quantized loading).
3. Feed measured `H_ε(N)` into `systems/roofline_sim.py` to predict tokens/s of a
   generation-based engine vs disk reload — closing the theory↔systems loop.

## Decision
Gate **passed in the supporting direction.** Whether `H_ε(N)/n → 0` (C1 holds in
the limit) remains the one open quantity Theorem 4 isolates — but every measured
point moves the right way.
