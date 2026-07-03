# The pseudo-model theorem: a PROVEN result (Door 4 made rigorous)

This document states and proves a genuine theorem — not a conjecture. It
is the rigorous core of "1T-capability in 12 GB": the counting side is
fully settled here; only the *finding* side (LCDL/Γ, Prop 5) stays open.
The theorem is a specialization of the Occam / compression bound
(Blumer, Ehrenfeucht, Haussler & Warmuth 1987; Vapnik–Chervonenkis
uniform convergence). Everything below is standard, correct, and
self-contained.

---

## 1. Setup and the distinguishing metric

Let f : X → Y be an arbitrary target function — "the trillion-parameter
teacher". Let Q be any distribution over inputs X (the query
distribution — what the user will actually ask). Fix a description
language L (e.g. bytes of a program/weights).

**Definition (distinguishing advantage).** For a responder g, the
single-query advantage of the best distinguisher is exactly the
disagreement probability
    d_Q(g, f) = Pr_{x∼Q}[ g(x) ≠ f(x) ].
A pseudo-model is ε-indistinguishable from f under Q iff d_Q(g,f) ≤ ε.
Over T independent queries the advantage is ≤ T·ε (union bound), so
matching to ε ≤ δ/T gives total advantage ≤ δ over a whole session.

---

## 2. Theorem 8 (Query-indistinguishable pseudo-model) — PROVEN

**Theorem.** Fix ε, δ ∈ (0,1) and a description budget of B bits. Suppose
a procedure draws m i.i.d. samples (x_i, f(x_i)), x_i ∼ Q, and returns
ANY g ∈ L with |g| ≤ B bits that is consistent, g(x_i) = f(x_i) for all
i. If
        m ≥ ( B·ln 2 + ln(1/δ) ) / ε
then, with probability ≥ 1 − δ over the sample,
        d_Q(g, f) ≤ ε.
That is: a B-bit responder that reproduces the teacher on enough sampled
queries is provably ε-indistinguishable from it under Q.

**Proof.** Call g "ε-bad" if d_Q(g,f) > ε. Fix one ε-bad g in advance.
Each sample independently agrees with f with probability
1 − d_Q(g,f) < 1 − ε, so the probability g survives (agrees on all m
samples) is < (1−ε)^m ≤ e^{−εm}. There are at most 2^B programs with
|g| ≤ B. By the union bound, the probability that *some* ε-bad program of
length ≤ B is consistent with all m samples is
        ≤ 2^B · e^{−εm}.
Setting 2^B e^{−εm} ≤ δ and solving: m ≥ (B ln2 + ln(1/δ)) / ε. Under
this m the returned consistent g cannot be ε-bad except with probability
≤ δ. ∎

**What this settles.** The *counting/statistics* objection to "1T in
12 GB" is dissolved for the indistinguishability contract: with B = 12 GB
the bound on m is finite and explicit. Counting (pigeonhole) is never
violated — most targets have NO short consistent g; the theorem only
promises that IF one is found, it generalizes. Finding it is the open
part (§5).

---

## 3. The formulas

Let b = log₂|Y| bits per answer. Write B = 8·(12·10⁹) = 9.6×10¹⁰ bits.

**(F1) Sample cost for ε-indistinguishability at 12 GB:**
    m*(ε,δ) = (B ln2 + ln(1/δ)) / ε
    e.g. ε = 10⁻³, δ = 10⁻⁶  ⇒  m* ≈ 6.6×10¹³ query–answer pairs.
(Information-theoretic sufficiency; a one-time teacher-query budget.)

**(F2) True effective size — the covering-number law.** The minimum
description length of an ε-indistinguishable responder is
    B*(ε) = ⌈ log₂ N(f, d_Q, ε) ⌉ bits,
where N(f, d_Q, ε) is the ε-covering number of the responder class under
the Q-weighted Hamming metric. This is the exact "how many bits do you
actually need" and it is governed by Q, not by the teacher's parameter
count:

  • Incompressible teacher (worst case): if f is a uniformly random
    function and Q is uniform on a domain of size D, then
    B*(ε) ≥ (1−H₂(ε))·D·b — linear in the domain. Pigeonhole intact:
    12 GB cannot fake it. (This is the negative control in E3.)

  • Concentrated / structured teacher (Door 2): if Q is effectively
    supported on K inputs, or f restricted to Q lies in a class of
    pseudo-dimension d, then
    B*(ε) ≤ min( K·b , O(d·log(1/ε)) ) — independent of teacher size.

**(F3) Session-level contract.** To be indistinguishable over T queries
with total advantage ≤ δ: require per-query ε ≤ δ/T, hence
    m ≥ (B ln2 + ln(1/δ))·T/δ.

**(F4) Channel form (link to §4 of CAPACITY_THEORY).** Writing
information per stored bit, the responder needs
    I(g ; f | Q) ≥ (1−H₂(ε))·H(f|Q)
bits — you must store at least the Q-conditional entropy of the teacher's
answers, discounted by the tolerated error. This is Door 2 (typical set)
and Door 3 (ε-slack) in one line, and it is the true floor.

---

## 4. The architecture that realizes it

A pseudo-model M within the 12 GB budget, decomposed by which door pays
for which part:

    M = ⟨ S : student weights,      (streamed, low-bit)   — Door 2 ⟩
        ⟨ R : fact store,           (lossless/BM25)       — Doors 2,3 ⟩
        ⟨ I : interpreter+library,  (programs)            — Kolmogorov ⟩

    response(x) = decode_M(x):
        p ← S(x)                       # cheap reasoner proposes
        if x needs facts: p ← p ⊕ R.retrieve(x)   # verbatim knowledge
        if x needs computation: p ← I.run(p, x)   # exact procedures
        return p

Bit budget (all already implemented on this branch):
    |S| via streaming.py + quant_cpu.py   (Door 2: typical, low-bit)
    |R| via factstore.py + lossless.py    (Doors 2–3: dense, ε-exact)
    |I| via induction.py                  (programs, Kolmogorov-dense)
    |S| + |R| + |I| ≤ 12 GB.

By (F2), M is ε-indistinguishable from a far larger teacher under Q
precisely when the teacher's Q-behavior is structured — which trained
models empirically are (that is why distillation works at all;
distillation is Theorem 8 with S the student and the training set the m
samples).

---

## 5. What is proven vs. open (no overclaim)

PROVEN (this document):
  • Theorem 8: a consistent B-bit responder is ε-indistinguishable after
    m*(ε,δ) samples. Counting objection dissolved for the ε-contract.
  • F2 negative control: for an incompressible teacher, B* is linear in
    the domain — 12 GB genuinely cannot fake a random 1T function.
    Pigeonhole holds exactly where its hypotheses hold.

OPEN (stated honestly, = the Fields-grade work):
  • Whether a *specific trained* teacher's Q-behavior has small B*
    (= Conjecture C / Theorem P): this is the LCDL question.
  • Whether SGD can FIND the short consistent g efficiently (= Γ,
    obstructed in general by Prop 5 / MCSP).

The theorem cleanly separates the two: **the storage is not the wall;
the search is.** That reframing is itself the contribution — it is why
chasing "new maths to beat counting" was the wrong target, and
"cheap learning of short indistinguishable responders" is the right one.

Verified empirically in docs/EXPERIMENTS.md, E3.
