# Capsule: research/tme token-multiplier stack

- **Hypothesis:** A deterministic kernel (measurement + directed memory + graph
  compilation + library learning) amplifies a proposer (Claude) — "more output per
  token" — without an LLM inside it.
- **Evidence:** every module has a passing `--selftest`. Measured, held-out
  validated: memory reuse x1.24 (engine cross-run x7 on repeat); igc graph reuse
  x1.33-2.0; abstraction transfer 0→2 on novel depth-3 tasks, MDL 8→7, hierarchical
  depth ladder 2→4→5. Negative results kept visible (self-reconfig x0.84 dropped;
  value-only routing trap).
- **Counter-evidence:** domain is a toy list/code DSL, NOT an LLM or real codebase;
  multipliers are VARIABLE and collapse to ~1 on novel/structureless work; the
  `adaptive_compute_probe` backend is a labelled synthetic stub (not a result).
- **Risk:** over-reading toy multipliers as general capability.
- **Confidence:** 80% the measured effects are real *within the toy domain*; 15%
  they transfer to real airllm edits without the (unbuilt) LLM-proposer bridge.
- **Decision:** HONEST research prototype, not a product. NOT a paradigm shift
  (author's own framing). Maturity: lab. Useful as a measurement/abstraction substrate.
