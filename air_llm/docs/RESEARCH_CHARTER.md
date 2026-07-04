# Research charter: the six missing elements, mapped

The correct program for "a 1T model fits in <12 GB without quality loss".
Below, each required element is mapped to what the repo already provides
and what is genuinely still missing. No overclaim: elements 1-5 have
rigorous partial forms; element 6 and the central obstacle are the open
empirical work.

1. PRECISE DEFINITION. "Without quality loss" = ε-indistinguishable under
   a query distribution Q (INDISTINGUISHABILITY.md, Def), NOT bit-exact
   (proven impossible by counting, CAPACITY_THEORY.md Thm 2). Benchmarks
   = the support of Q; model family = fixed teacher f. STATUS: defined.

2. NEW REPRESENTATION. The pseudo-model M = <S reasoner, R fact store,
   I interpreter+library> (INDISTINGUISHABILITY.md §4), implementation-
   independent (defined by its input->output behaviour). STATUS: defined,
   implemented (streaming/quant_cpu/factstore/induction).

3. TRANSFORMATION ALGORITHM. Transformer -> M is distillation (fit S on
   m samples from Q) + externalization (facts->R, procedures->I).
   Complexity: m*(ε,δ) samples (F1); Kt-bounded program search for I
   (Thm 10). STATUS: algorithm specified; end-to-end distillation
   pipeline NOT yet in repo.

4. EQUIVALENCE PROOF. Theorem 8: a consistent B-bit M is
   ε-indistinguishable from f under Q w.h.p. Error explicitly bounded
   (d_Q <= ε). STATUS: proven.

5. THEORETICAL BOUNDS. Max size B*(ε) = log N(f,d_Q,ε) (F2); floor
   H(f|S,Q) (Thm 9); reconstruction/query time via streaming
   (Thm 3/5). STATUS: bounds proven; the teacher-specific value of B*
   is the open quantity.

6. REPRODUCIBLE EXPERIMENTS + THE CENTRAL OBSTACLE. Does exploitable
   redundancy GROW with model size? If yes, large models compress
   super-linearly and the program succeeds; if no, most parameters are
   essential. THIS IS THE DECIDING MEASUREMENT. Incremental strategy
   (small -> large -> trend -> limits) per the charter. STATUS: E12
   below takes the first step; scaling to real 1B-70B models with a
   mutual-information estimator is the remaining work.

## The deciding measurement (E12, first step)

Redundancy present in TRAINED weights but absent in RANDOM weights, as a
function of model size, is a lower bound on exploitable structure. If
this gap widens with size, the compressibility thesis has empirical
support; if it stays flat, it does not. Measured by the repo's own
lossless codec (trained vs random bits/param), across a size sweep.
