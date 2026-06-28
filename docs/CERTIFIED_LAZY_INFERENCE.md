# Certified Lazy Inference — a lossless software mechanism (and its honest limits)

> Companion to `THEOREM_KOLMOGOROV.md` / `PROOFS_KOLMOGOROV.md`. Proposes the one
> *lossless, software* mechanism that does not contradict Theorem 4, states its
> correctness theorem (proven) and average-cost characterization, and reports the
> **sobering empirical probe** that bounds how much it can actually buy.

## 1. Idea

Theorem 4 (`S*_0 = Θ(H_0)`) is **worst-case**. The escape — if any — is
*average-case over natural inputs*, exploited **losslessly** by computing only as
much of the model as needed to *prove* the exact output token, then stopping.
Inference becomes adaptive evidence-gathering, not fixed computation.

## 2. Algorithm (greedy decoding)

```
ℓ̂, B ← partial logit estimate and a valid bound;  budget b ← 0
repeat:
    read next weight increment (layer / column / expert); update ℓ̂(b)
    B(b) ← valid upper bound on ‖ℓ − ℓ̂(b)‖_∞      # monotonically decreasing
    if  ℓ̂(b)_(1) − ℓ̂(b)_(2) > 2·B(b):            # top-2 gap of partial logits
        return argmax ℓ̂(b)                        # CERTIFIED EXACT — stop early
until b = full model
return argmax ℓ̂(full)                             # B = 0, trivially exact
```

## 3. Theorems

> **Theorem L1 (lossless certificate).** Let `ℓ` be the true logits, `ℓ̂` a
> partial estimate, `B ≥ ‖ℓ − ℓ̂‖_∞` valid. If `ℓ̂_(1) − ℓ̂_(2) > 2B` then
> `argmax ℓ = argmax ℓ̂`.

*Proof.* For `i*=argmax ℓ̂`: `ℓ_{i*} ≥ ℓ̂_(1) − B`. For `j≠i*`:
`ℓ_j ≤ ℓ̂_(2) + B`. The hypothesis gives `ℓ̂_(1) − B > ℓ̂_(2) + B`, so
`ℓ_{i*} > ℓ_j` for all `j`. ∎

**Corollary (global exactness).** The algorithm always returns the exact
`argmax ℓ`: either the certificate fires (exact by L1) or the whole model is read
(`B=0`). **Lossless regardless of how loose the bound `B` is** — bound tightness
governs *speed*, never *correctness*.

> **Theorem L2 (average cost).** With `C(x) = min{ b : B(b) < margin(x)/2 }` and
> `margin(x) = ℓ_(1) − ℓ_(2)`, the expected lossless cost is
> `E_x[C] = Σ_b Pr[ B(b) ≥ margin(x)/2 ]`. Worst case `C = full = H_0` (consistent
> with Theorem 4); confident inputs (large margin) certify at small `b`.

L2 makes the **duality** precise: *expected lossless cost is governed by the
model's confidence (margin / entropy) on the input distribution.* Low perplexity
⇒ large margins ⇒ cheap certification.

## 4. Empirical probe (the sobering part) — GPT-2 / WikiText-2

`kolmogorov/experiments/lazy_inference_probe.py`, 2048 tokens, logit-lens depth
oracle (the *best possible* depth-axis early stop, not even requiring a valid
bound):

| quantity | value |
|---|---|
| depth-lock fraction (mean / median / p90) | **0.96 / 1.00 / 1.00** |
| tokens locked by half-depth | **2.5%** |
| tokens locked by 75%-depth | 6% |
| median logit margin | 0.80 |
| median entropy | 4.0 nats (~55 effective tokens) |

**Reading.** On GPT-2 the exact argmax depends on the *late* layers: even an
idealized depth-axis oracle saves only ~2.5% of tokens at half depth. The
**depth-axis instantiation of lazy inference is refuted** for this model. This is
exactly L2's "uncertain regime": GPT-2's high entropy (4 nats) ⇒ small margins ⇒
expensive certification. The mechanism of the duality is *confirmed*; GPT-2 sits
on its unfavourable side.

## 4b. The crack OPENS with scale (tests Theorem L2)

`kolmogorov/experiments/lazy_inference_scaling.py` — the depth-lock oracle across
the Pythia suite, WikiText-2:

| N | mean lock-depth (≈ E[C]/full) | locked by half-depth | median margin | entropy (nats) |
|---:|---:|---:|---:|---:|
| 70M  | 0.975 | 0.3%  | 0.79 | 4.27 |
| 160M | 0.940 | 1.6%  | 0.86 | 3.93 |
| 410M | 0.844 | 5.9%  | 0.97 | 3.16 |
| 1.4B | **0.803** | **17.4%** | 1.04 | 2.86 |

**Monotone, accelerating, in L2's predicted direction.** Over 20× scale the depth
oracle cost falls 0.975→0.803, margin rises, entropy falls, and the fraction of
tokens locked by half-depth grows ~58× (0.3%→17.4%). This is the **first positive
evidence** that the lossless crack widens with model quality — exactly the L2
duality (confident models certify cheaply).

Honest limits: it is a depth-axis *oracle* (logit lens, optimistic upper bound on
savings, not a valid certificate); even at 1.4B the average token still needs 80%
of depth; and this is extrapolation toward — not a measurement at — trillion
scale. The direction is unambiguous; the magnitude at scale is unproven.

## 4c. The MoE (expert) axis — native lossless sparsity, but not prefetchable

`kolmogorov/report/moe_routing_results.json` — Switch-base-8, encoder routers:

- **Active fraction per token = 12.5% (top-1), EXACT/lossless by construction.**
  Streaming only the routed expert is the model's true computation, so a native
  MoE gives a lossless sparsity lever for free — the single biggest structural
  win for "1T on a Pi" without any approximation.
- **Router is near-uniform**: top-1 expert probability median 0.28 (uniform =
  0.125), routing entropy 1.99/2.08; 0% of tokens routed with confidence > 0.5.
  Consequence: you **cannot cheaply *prefetch* the routed expert** before running
  the router — the choice genuinely depends on the full router. But the router is
  itself cheap, so losslessness holds (compute router exactly → stream one
  expert); only *anticipatory* prefetch is denied.

Net: native MoE is the cleanest lossless lever (exact sparsity); contextual
expert-*prediction* is not lossless-cheap on this model.

## 5. Honest verdict

- **Worst case:** Theorem 4 stands — lossless dense inference cannot beat `H_0`.
- **Algorithm:** Certified Lazy Inference is *correct and lossless* (Thm L1),
  unconditionally. That part is solid.
- **Payoff:** governed by confidence (Thm L2). **Measured on the only model we can
  run (GPT-2): small** — argmax locks late, the depth axis is essentially dead.
- **Scale (now measured, §4b):** the depth-lock oracle cost *decreases
  monotonically* with N (0.975→0.803 over 20×) and tokens-locked-by-half grows
  ~58×. The L2 duality is **empirically supported**: the lossless crack widens
  with model quality. Still an oracle/upper-bound and still 0.80 at 1.4B —
  direction proven, trillion-scale magnitude not.
- **MoE axis (now measured, §4c):** native top-1 routing gives an *exact* 12.5%
  active fraction — a real lossless sparsity lever — but routing is near-uniform,
  so the expert is not cheaply *prefetchable*.
- **Still open, heavy burden:** the brick — a bound `B(b)` that is valid,
  fast-decaying, and `o(H_0)`-cheap — turning the oracle trend into a real
  certified speedup; and whether the trend continues to trillion scale.

## 6. Where this leaves the dream

A *lossless* software paradigm shift for *dense interactive* 1T on a Pi is **not
demonstrated**, and the worst-case theory says it cannot exist in general. The
honest residue: lossless speedups are possible exactly to the extent a model is
*confident*, they grow (conjecturally) with scale, and they are *certified*, never
guessed. That is a real, narrow, honest opening — not the paradigm shift, but the
only software direction that is simultaneously lossless and not ruled out.

## 7. References
Leviathan et al. 2023 / Chen et al. 2023 (speculative decoding — exactness via
verification); Schuster et al. 2022 (CALM, confident adaptive early exit, lossy);
nostalgebraist (logit lens); Gehr et al. / Gowal et al. (interval bound
propagation — candidate for `B`); this project, `PROOFS_KOLMOGOROV.md` Thm 4.
