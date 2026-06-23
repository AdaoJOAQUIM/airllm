# Capsule: stress test of the deterministic kernel (CPU, no GPU/LLM)

- **Hypothesis:** the kernel is robust and scales; claims survive being pushed.
- **Evidence (measured, `research/tme/stress_test.py`):**
  - S1 search wall: full scan = |ops|^depth. MAXD 1→4 = 22 / 506 / 11,154 / 245,410
    candidates (~5-6e5 cand/s). **Brute search is the hard limit** depth≥4 — exactly
    what abstraction/proposer-pruning exists to avoid.
  - S2 vocabulary: cost grows ~|ops|^MAXD (+100 macros → 15,006 candidates). A large
    DSL **without a pruning proposer is intractable** — confirms the design constraint.
  - S3 persistence: 1,600 signatures / 5,000 patterns, bounded by mem_cap=6;
    save 11ms/194KB, load 4ms, round-trip integrity OK. Scales fine.
  - S4 nested macros: resolution OK to depth 2,000 (not guarded but no shallow break).
  - S6 determinism: paradigm_report & curriculum_report reproducible.
- **Counter-evidence / BUG FOUND:** S5 — `igc.compile_intention` **silently accepts
  malformed input**: `x=nope(input)` (unknown op → deferred KeyError at execution),
  `=double(input)` (empty name), `""` (empty graph). Only `x=` and `sum()` error.
- **Risk:** unknown-op intentions fail late/opaquely instead of at parse time.
- **Confidence:** 90% the scaling walls are as measured; 99% the igc silent-accept
  bug is real.
- **Decision:** kernel scaling/robustness mostly sound within design limits; **harden
  `igc.compile_intention`** to reject unknown ops / empty names at parse time. Brute
  search depth and big-vocabulary cost are inherent → must stay proposer-pruned.
