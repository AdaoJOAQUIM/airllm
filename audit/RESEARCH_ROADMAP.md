# RESEARCH_ROADMAP — from offloading to cognitive compression

Phase after the audit (a "Research Gap Finder"). Built under the same rule as the
audit: truth/tokens, and proven / plausible / speculative are kept distinct. No code
yet — this is the plan that gates code.

Central question (reframed from "1T on a Raspberry Pi"):
> Can a giant model's **capability** be obtained without explicitly storing **all**
> its parameters — i.e. by exploiting their structure or replacing them with a
> generator — and does the idea survive the audit's level of scrutiny?

## 0. What the audit established (anchor)
AirLLM relocates storage (disk→VRAM, layer-by-layer) + standard quantization. It
treats weights as **opaque data**; it compresses *where weights live*, not *how much
information they carry*. So the gap is structural: it exploits none of the known
redundancy of trained networks.

## 1. Fundamental limits — what is indispensable vs redundant

- **Indispensable:** the *function* the network computes. Its true size is lower-
  bounded by the description length (≈ Kolmogorov complexity) of that function — a
  hard floor. "Infinite capability in a few GB" violates this and is **refuted a
  priori** absent extraordinary evidence.
- **Empirically redundant (large fraction):** 16→4-bit quant ≈ 4× lossless-ish;
  pruning often 50-90% sparsity; finetuning deltas are low-rank (LoRA); distillation
  matches larger models on tasks. So the *capability* needs far fewer bits than the
  *stored* parameters — the floor is well below current footprints.
- **AirLLM's miss:** it captures *none* of this structurally. The research gap =
  replace "store + stream all weights" with "store a compact representation/generator
  and reconstruct-or-skip weights on demand."

Why AirLLM needs the real weights today: it runs the exact learned function with no
model of its structure. Every brick below is a way to *stop* needing all of them.

## 2. Missing bricks — honest state of each

| Brick | What it buys | Maturity / verdict |
|---|---|---|
| **Extreme PTQ (codebook, <2-bit: QuIP#, AQLM)** | smaller stored model, post-hoc | **proven→frontier**; 3-4bit solid, ~2bit promising. Highest near-term leverage |
| **Low-rank / sparse factorization of base weights** (W≈UV+S) | fewer bits at equal quality | proven for *deltas* (LoRA); base-weight at quality = **research** |
| **Sparse activation / dynamic MoE + expert streaming** | load only active experts, not all layers | **proven** (Mixtral); expert-offload = natural AirLLM successor |
| **Retrieval-augmented weights** (kNN-LM, RETRO) | move *facts* out of parameters | **proven for knowledge**; reasoning stays in weights (partial) |
| **HyperNetworks (a small net generates the big net's weights)** | weights become *implicit* (a function) | **speculative at LLM scale**; the true paradigm candidate, low evidence |
| **Activation-based reconstruction** (sketching/random features) | approximate a layer without its full matrix | **speculative**, high risk |
| Advanced cross-layer weight sharing (ALBERT/UT) | fewer unique params | proven at moderate scale; must be trained in |
| Learned optimizer / low-rank manifolds of training | tangential to storage | low relevance to the compression question — **pruned** |

## 3. Possible vs impossible map (as stated, honestly)

| Claim | State |
|---|---|
| 405B in 8GB VRAM | **already** (AirLLM: layer streaming + quant) |
| 1T in ~100-250GB (4-bit + streaming) | **plausible** with engineering |
| 1T with sparse expert streaming, lower latency than dense | **plausible, untested here** |
| 1T on a Raspberry Pi at usable speed | **not demonstrated** — even 2-bit (~250GB) exceeds Pi storage/RAM; latency catastrophic |
| Replace weights by a generative function at LLM scale | **research, unproven** |
| "Infinite capacity in a few GB" | **refuted** (violates the description-length floor) |

## 4. Falsifiable hypotheses with go/no-go (the actual roadmap)

Ranked by leverage × testability. "Cheap" = small model, often CPU/single-GPU.

**H1 — Extreme weight compression preserves capability (near-term, cheap).**
- Experiment: quantize a small open model to ~2-bit (codebook/QuIP#-style) vs 4-bit.
- Metric: bits/param **and** perplexity/task-accuracy delta vs fp16.
- Expected: <2-bit at ≤ small quality loss → 2×+ smaller AirLLM footprint.
- **Go/no-go:** GO if quality loss within budget at >2× further compression; else NO.

**H2 — Expert streaming beats dense layer streaming (near-term).**
- Experiment: at fixed VRAM, compare tokens/sec for dense-layer-streamed vs
  MoE-expert-streamed inference (only active experts loaded per token).
- Metric: tokens/sec @ memory; quality held constant.
- Expected: higher throughput at equal memory (load less, not everything).
- **Go/no-go:** GO if throughput gain is significant; this is AirLLM's natural v2.

**H3 — Weights are partly *generatable* (the paradigm probe, high-risk, cheap to falsify).**
- Experiment: train a small hypernetwork to regenerate **one** transformer layer's
  weights (or match its input→output function) from a low-dim code; substitute it.
- Metric: reconstruction error + downstream perplexity when that layer is generated.
- Expected (honest): likely degrades; but the **one-layer** test is the cheap
  decisive signal on whether "weights as a function" is viable at all.
- **Go/no-go:** GO to scale only if one layer holds quality within budget; else STOP
  and document (scientific pruning) — this is how AirLLM would be made *obsolete*, or
  the idea is killed cheaply.

**H4 — Knowledge externalization shrinks the parametric model (medium).**
- Experiment: retrieval-augment a small model; measure how much parametric size can
  drop at equal task accuracy when facts are retrieved, not stored.
- Metric: params vs accuracy with/without retrieval.
- **Go/no-go:** GO if a meaningful parametric reduction holds at equal accuracy.

## Verdict on the "minimal innovation"
There is **no single magic brick**. The defensible minimal step that turns AirLLM
from *offloading* toward *cognitive compression* is **structural, not magical**:
combine **load-less** (sparse/expert streaming, H2) with **store-smaller** (extreme
learned compression, H1). The genuine *paradigm* candidate — replacing weights with
a generator (H3) — is high-reward but currently low-evidence; the roadmap's value is
that it specifies the **cheap one-layer experiment that would give the first honest
signal**, and the go/no-go that kills it fast if it doesn't.

Constraint kept from the whole programme: build only what an experiment proves > the
baseline. The description-length floor is real; the win is in removing *redundancy*,
not in defying information theory.
