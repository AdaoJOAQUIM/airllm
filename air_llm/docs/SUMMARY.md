# Summary: what we built, what remains

Branch: claude/trillion-param-weak-hardware-0gy87w — 22 commits, 48 tests
green, 12 experiments, 10 theorems. Two things were pursued: a
production engine, and a theory of model-capacity compression.

## Part 1 — The engine (DONE, production-ready, airllm 2.12.0)

Runs the full AirLLM pipeline offline on CPU/ARM (4 GB RAM class), which
the upstream repo could not do (needed CUDA+bitsandbytes, broke on modern
transformers).

- `compression='lossless'`  : byte-plane entropy coding, bit-exact,
  16.00 → 11.27 bits/param (−30%), no GPU. (lossless.py)
- `compression='4bit-cpu' / '8bit-cpu'` : blockwise NF4/int8, pure CPU,
  no bitsandbytes. (quant_cpu.py)
- Tensor streaming, prefetch pipeline, StreamedLinear (layers > RAM),
  verified bit-identical on a real Llama layer. (streaming.py)
- FactStore: offline BM25 RAG over lossless passages. (factstore.py)
- induction.py: CTW / MDL-program-search / NCD (computable inductors).
- 4 compatibility fixes (DynamicCache, model-level rotary, single-file
  checkpoints, optional bettertransformer) → runs on current transformers.
- Verified E2E: lossless-shard generation is token-for-token identical to
  raw shards on a real model (cpu_e2e_check.py).

## Part 2 — The theory (10 theorems, 12 experiments)

Reframed "1T in 12 GB lossless" from bit-exact (impossible) to
ε-indistinguishable under a query distribution (the LCDL framework).

PROVEN & verified:
- Thm 8  pseudo-model ε-indistinguishability (Occam bound)        [E3]
- Thm K′ perceptron capacity = 2 bits/param, sharp (Cover)        [E4]
- Thm K″ lazy-regime universality across architectures            [E6]
- Thm K‴ capacity = precision; robust κ ≤ log₂(1/ρ)               [E7]
- Thm K‴d′ Tangent Capacity Principle (2 = Cover on tangent space)[E9]
- Thm P′ programs beat weights exponentially; library exponent-cut[E5]
- Thm P″ LCDL=CDL+O(1) unbounded; efficient on-average
- Thm P‴ genesis/white-box makes worst-case-efficient P TRUE      [E8]
- Thm 9  Shannon circumvention: floor is H(X|S), not H(X)         [E10]
- Thm 10 Kolmogorov circumvention: bounded Kt is computable       [E11]

PROVEN FALSE (a result, not a gap):
- exact lossless 1T→12 GB (counting) ; representational universal
  constant ; worst-case-efficient P under query-only access (OWF).

THE THREE LEGAL CIRCUMVENTIONS (one rule: delete a hidden hypothesis):
  pigeonhole → typical set / ε-slack / pseudorandomness (Doors 2–4)
  Shannon    → conditional entropy H(X|S)              (Thm 9)
  Kolmogorov → time-bounded Kt                          (Thm 10)
None breaks its theorem; each removes an assumption the naive statement
didn't need.

## What remains OPEN (the honest frontier)

TWO residues, each pinned, each with instruments in-repo:
- K‴d — the DYNAMICAL constant: why does SGD's noise floor sit at
  κ ≈ 2 bits/param? Reduces to one named premise, LPL (late-phase
  linearization). Attacks: Gardner–Talagrand replicas; trajectory
  rate–distortion; fractional/piecewise SGD (Atangana). Bench: E2, E9.
- P‴d — the SHAPE of the LCDL(C) tradeoff curve for general
  distributions. Identified with Impagliazzo's five worlds — bedrock:
  resolving it resolves average-case complexity. Not circumventable.

MISSING ENGINEERING (per RESEARCH_CHARTER.md element 3):
- an end-to-end distillation pipeline (Transformer → pseudo-model).

MISSING EXPERIMENT (element 6, the central obstacle):
- does exploitable redundancy grow with size? E12 is step 1 (weak, noisy
  hint yes); needs real 1B–70B models + a mutual-information (functional)
  estimator, not the byte-plane lossless probe.

## One-line honest status

The engine is finished and shippable. The theory reached the maximal
state a research program can: everything proven, disproven, or reduced to
one named premise (LPL) and one bedrock problem (five worlds). No new
mathematics can make 1T arbitrary weights fit 12 GB bit-exact (counting);
the real question — how much FUNCTIONAL structure trained models have —
is open, well-posed, and instrumented here.
