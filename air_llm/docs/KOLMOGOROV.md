# Theorem 10: the legal circumvention of Kolmogorov (bounded Kt)

Kolmogorov complexity K(x) = min{ |p| : U(p) = x } is UNCOMPUTABLE and
not approximable to within any bound — it inherits the halting problem
(Turing 1936). The naive reading: "the shortest description is
forbidden, so description-length methods are unusable." The hidden
hypothesis is **unbounded computation time**. Delete it.

## Statement

Define Levin's time-bounded complexity (Levin 1973):
        Kt(x) = min_p [ |p| + log₂ time(p → x) ],
the shortest program weighted by the log of its runtime. Then:
 (i) Kt(x) is COMPUTABLE (total): it is found by dovetailed Levin search,
     which halts for every x.
 (ii) Kt(x) ≥ K(x), and Kt(x) ≤ K(x) + log₂(fastest runtime) — it equals
     K up to the unavoidable log-time term.
 (iii) The associated universal search finds a program of Kt-value ≤ t
     after O(2^t) steps — computable, unlike K.

## Proof

(i) Enumerate all (program p, step budget s) pairs in increasing order of
|p| + log₂ s. For each, run p for s steps; if it outputs x, record
|p| + log₂ s. The first success along this order is Kt(x). The pair
(p*, time(p*)) realizing the minimum has finite |p*| + log₂ time(p*), so
it is reached in finitely many enumeration steps — the procedure halts.
No halting oracle is consulted; runtimes are capped, never awaited. ∎
(ii) Adding a nonnegative log-time term to |p| gives Kt ≥ K; taking the
K-minimizer p* with its own runtime gives the upper bound. ∎
(iii) The dovetail spends a 2^{−(|p|+log s)} fraction of steps on each
pair, so any pair of Kt-value ≤ t gets a constant share within O(2^t)
total steps (Levin's optimality). ∎

## Why this circumvents Kolmogorov (without breaking it)

K(x) is still uncomputable — Theorem 10 does not touch that. It observes
that the *usable* quantity for prediction, learning and compression was
never K but Kt: every real compressor, every real learner runs in
bounded time, so it already computes a bounded-Kt surrogate. The
impossibility applied to an idealization no algorithm ever needed. This
is exactly the third door of Solomonoff induction (THEORY.md, Theorem 7)
made into a standalone circumvention parallel to Shannon (Theorem 9) and
pigeonhole (CAPACITY_THEORY.md §6).

## Consequence for the project

The interpreter brick I in the pseudo-model architecture
(INDISTINGUISHABILITY.md §4) is a bounded-Kt coder: induction.py's
mdl_program_search enumerates programs in exactly the |p| + (implicit
time) order and returns the shortest one found within budget — computable
Kolmogorov compression of the algorithmic part of capability. The three
circumventions now stand together, each deleting one hypothesis:

  pigeonhole → typical set / ε-slack / pseudorandomness (Doors 2–4)
  Shannon    → conditional entropy H(X|S)              (Theorem 9)
  Kolmogorov → time-bounded Kt                          (Theorem 10)

Verified: E11 (examples/kt_complexity_demo.py) — bounded-Kt search
returns the shortest generating program for structured sequences and
HALTS on every input, where unbounded K search would not.
