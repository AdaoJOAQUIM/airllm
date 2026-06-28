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

## 5. Honest verdict

- **Worst case:** Theorem 4 stands — lossless dense inference cannot beat `H_0`.
- **Algorithm:** Certified Lazy Inference is *correct and lossless* (Thm L1),
  unconditionally. That part is solid.
- **Payoff:** governed by confidence (Thm L2). **Measured on the only model we can
  run (GPT-2): small** — argmax locks late, the depth axis is essentially dead.
- **Open, with a now-heavy burden of proof:**
  1. *Other axes* (width/columns, MoE experts) may lock earlier than depth.
  2. *Scale:* perplexity falls with size (our Stage 1), so by L2 the cheap regime
     should open up for large (confident) models. **This is the one hopeful,
     testable thread — and it is unproven; the single data point we have says
     "small."**
  3. The open brick: a bound `B(b)` that is valid, fast-decaying, and `o(H_0)`-cheap.

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
