# Theorem 9: the legal circumvention of Shannon (shared-context coding)

Shannon's source coding theorem is often quoted as an impossibility:
"to convey X you need at least H(X) bits, no code does better." True —
under one hypothesis the naive statement hides: **the decoder has no
side information.** Delete that hypothesis. What remains is a theorem,
and it is Shannon's own conditional version.

## Statement

Let encoder and decoder share side information S (a prior, a pretrained
model, a shared context). The minimum expected code length to convey X
is the conditional entropy
        L*(X | S) = H(X | S) = H(X, S) − H(S),
and H(X | S) ≤ H(X), with equality iff X ⊥ S. For a stream of queries,
the amortized cost is the conditional entropy rate h(X | S), which for
structured sources → 0.

## Proof

Achievability: conditional arithmetic coding with the shared model
P(X | S) attains expected length < H(X | S) + 1 (Kraft + the
information inequality E[−log P(X|S)] ≥ H(X|S) with equality at the true
model). Converse: no uniquely decodable code conditioned on S beats
H(X | S) (Cover & Thomas, Elements of Information Theory, Thm 5.3.1
conditioned on S). Monotonicity H(X|S) ≤ H(X) is "conditioning reduces
entropy" (Jensen on the concave −log). ∎

## Why this "circumvents" Shannon (and why it does not break it)

The bound H(X) that seems to forbid cheap storage is the floor for a
decoder that knows *nothing*. A pretrained student S, a fact store, an
interpreter — each is shared side information. The operative floor is
therefore H(f | S), NOT H(f). Shannon is not violated anywhere; the
naive impossibility simply used the wrong entropy. Formally this unifies
three known "circumventions", each deleting one hypothesis of the naive
statement:

  • delete "no side information" → conditional entropy H(X|S)
    (Shannon 1948; Slepian–Wolf 1973 for distributed S). THIS theorem.
  • delete "exact" → rate–distortion R(D): H drops to R(D) < H
    (Shannon 1959). Already used as the ε-slack of Theorem 8 / F2.
  • delete "ensemble average" → Kolmogorov complexity of the individual
    sequence, K(x) (Solomonoff/Kolmogorov). Already the induction.py core.

## Formula and consequence for 1T-in-12GB

Bits a pseudo-model must store to answer under Q, refined from
INDISTINGUISHABILITY.md F4:
        B*(ε) ≥ (1 − H₂(ε)) · H( f | S, Q ),
where S is everything the device already holds (pretrained reasoner +
fact store + interpreter). The trillion-parameter teacher's answers are
overwhelmingly *predictable from S* on natural Q — so H(f | S, Q) is
small even though H(f) is enormous. That gap, quantified, is the whole
game: distillation and RAG are conditional-entropy coders, and their
success is this theorem, not a violation of Shannon's.

Verified: E10 (examples/conditional_coding_demo.py) — a structured
source costs ~1 bit/symbol marginally (H, order-0) but ~0 under a shared
sequential model (H(X|past)), measured with the repo's CTW coder.
