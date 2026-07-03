# Proven theorems: the rigorous cores of K and P

Full Theorem K (universal κ across all architectures) and full Theorem P
(polylog LCDL for general algorithmic classes) are open — and this
document does not pretend otherwise. What follows are their *proven
restricted cores*, obtained by the only legal circumvention of an
impossibility (CAPACITY_THEORY.md §6): deleting one hypothesis. K′
deletes "all architectures" and proves the constant exactly for one.
P′ deletes "general task classes" and proves the separation for
compositional families. These are precisely stage 1 of the two laureate
routes (§5), executed. K′ is Cover's theorem (1965), presented
self-contained; P′ is an elementary original assembly.

---

## Theorem K′ (perceptron capacity: κ = 2 bits/parameter, exactly)

**Setting.** Points x₁,…,x_m ∈ ℝⁿ in general position (every subset of
≤ n points is linearly independent). Predictors: homogeneous linear
threshold units f_w(x) = sign(w·x), n real parameters. A *dichotomy* is
an assignment of ±1 labels to the m points; it is *realizable* if some w
produces it. Let C(m, n) be the number of realizable dichotomies.

**Lemma K′.1 (Cover's recurrence).** C(m+1, n) = C(m, n) + C(m, n−1).

*Proof.* Add the point x_{m+1}. Every realizable dichotomy d of the
first m points extends to at least one realizable dichotomy of m+1
points (take any w realizing d; sign(w·x_{m+1}) gives one extension).
It extends to *both* labelings of x_{m+1} exactly when some w realizing
d has w·x_{m+1} = 0 — then perturbing w slightly in ±x_{m+1} directions
realizes both signs without disturbing the m strict inequalities. Such
w live in the hyperplane H = x_{m+1}^⊥ ≅ ℝ^{n−1}, and realizing d
within H is equivalent to separating the projections of x₁,…,x_m onto
H, which are in general position in ℝ^{n−1} (general position of the
originals with x_{m+1}). So the number of doubly-extendable dichotomies
is C(m, n−1), and C(m+1, n) = C(m, n) + C(m, n−1). ∎

**Lemma K′.2 (closed form).** With base cases C(1, n) = 2 (n ≥ 1) and
C(m, 1) = 2,
    C(m, n) = 2 · Σ_{k=0}^{n−1} binom(m−1, k).
*Proof.* Induction on m via Pascal's rule binom(m,k) =
binom(m−1,k) + binom(m−1,k−1) applied to the recurrence of K′.1. ∎

**Theorem K′ (sharp storage threshold at 2 bits/param).** Assign the m
points i.i.d. uniform random ±1 labels ("m bits of pure knowledge").
Let P_store(m, n) = C(m, n)/2^m be the probability that the perceptron
can recall all m labels exactly. Then:
 (i) P_store(2n, n) = 1/2 exactly;
 (ii) for every δ > 0, P_store((2−δ)n, n) → 1 and
      P_store((2+δ)n, n) → 0 exponentially fast as n → ∞.
Hence the perceptron stores 2n random bits with n parameters — and not
one bit more, asymptotically: **κ_perceptron = 2 bits/parameter**.

*Proof.* P_store(m, n) = 2^{1−m} Σ_{k≤n−1} binom(m−1, k) =
Pr[ Bin(m−1, ½) ≤ n−1 ].
(i) For m = 2n the symmetry k ↔ (2n−1)−k pairs {0,…,n−1} with
{n,…,2n−1}, so the sum is exactly half of 2^{2n−1}, giving ½.
(ii) For m = (2−δ)n, the mean of Bin(m−1, ½) is (m−1)/2 ≈ (1−δ/2)n,
which is below n−1 by ~δn/2; Hoeffding gives
P_store ≥ 1 − exp(−δ²n/8(1+o(1))). For m = (2+δ)n, symmetric. ∎

**Corollary K′.3 (no learning gap here).** Realizing a realizable
dichotomy is a linear feasibility problem (find w with
y_i(w·x_i) ≥ 1), solvable in polynomial time by LP. For the
perceptron, storage capacity AND efficient search coexist: Γ ≈ 1.
The learning wall of Proposition 5 is a property of *richer* classes,
not of storage itself.

**Scope.** This proves Conjecture A exactly for one architecture — the
anchor of the universality question. Gardner (1988) recovers α_c = 2 by
replica methods; Talagrand rigorized parts of that program. Our bench
measures 2.10 (MLP, EXPERIMENTS.md E2) and Allen-Zhu & Li report ~2
(transformers): the conjecture is that K′'s constant is universal.
Empirical check of the sharp threshold: E4.

---

## Theorem P′ (library amortization beats weight memorization)

**Setting.** A finite primitive library L = {g₁,…,g_ℓ}, total encoding
B_L bits. A *compositional family* is a set of tasks D₁,…,D_T over a
domain X, |X| = 2ⁿ, with b-bit outputs, where each D_t is computed
exactly by a chain composition g_{i_k} ∘ ⋯ ∘ g_{i_1} of length ≤ k.

**Theorem P′.** For every compositional family:
 (a) [amortized description] The whole family is encoded in
     CDL(family) ≤ B_L + T · k⌈log₂(ℓ+1)⌉ + O(log T) bits —
     a *marginal cost per task of k⌈log₂(ℓ+1)⌉ bits, independent of
     |X|* (the ℓ+1 alphabet includes a stop symbol).
 (b) [weight floor] Any storage of the same behavior as memorized
     input–output facts at rate κ bits/param (the Theorem K regime,
     Lemma 2 of CAPACITY_THEORY.md) needs at least |X|·b·(1−o(1))/κ
     parameters per task when the behavior table is incompressible as a
     table. The ratio of (b) to (a) is Ω(2ⁿ b / (k log ℓ)) —
     **exponential in n**.
 (c) [search, and why libraries help] Enumerating chains in increasing
     length finds each task's program after at most
     Σ_{j≤k} ℓ^j ≤ 2ℓ^k candidate evaluations. If two discovered
     programs h₁, h₂ are added to the library (ℓ′ = ℓ+2) and a later
     task equals h₂ ∘ h₁, its search cost drops from Θ(ℓ^k) to at most
     2ℓ′² — the *exponent* falls from k to 2. Library growth buys
     exponent reduction, not constant-factor reduction.

*Proof.* (a) Emit the library once (B_L bits), then per task the index
sequence i₁,…,i_j, j ≤ k, over the (ℓ+1)-letter alphabet with stop
symbol: ≤ k⌈log₂(ℓ+1)⌉ bits; O(log T) delimits the count. Correctness:
the chain determines the task's output on every x ∈ X. (b) Immediate
from Lemma 2 (Fano/rate–distortion floor) applied per task. (c) The
number of chains of length ≤ k over ℓ primitives is Σ_{j≤k} ℓ^j; the
enumeration in nondecreasing j is exhaustive, so the target chain is
met within that budget. For the enlarged library, chains of length ≤ 2
over ℓ′ primitives number ≤ ℓ′ + ℓ′² ≤ 2ℓ′². ∎

**Scope.** P′ is Theorem P with "algorithmic structure" instantiated as
bounded compositionality — the restricted theorem the route in §5
designates as stage 1. What stays open is the general-Kt version, which
collides with MCSP-type hardness (Prop 5's wall). Note what P′ already
settles: *whenever behavior has compositional structure, programs beat
weights by an exponential factor, provably* — the formal content of
"knowledge as programs". Empirical check: E5.

---

## Theorem K′′ (architecture-universality in the lazy regime)

*Hypothesis deleted from full K: feature learning. What remains: any
architecture trained in the linearized (lazy/NTK) regime — and there,
universality is a theorem.*

**Setting.** Take ANY differentiable architecture f_θ with P parameters,
trained in the linearized regime around a centered initialization θ₀
(f_{θ₀} ≡ 0, achievable by the standard antisymmetric-duplication
construction): f_θ(x) ≈ ∇_θ f_{θ₀}(x) · (θ − θ₀). Decisions by sign.
Assume the gradient features φ(x_i) = ∇_θ f_{θ₀}(x_i) ∈ ℝ^P of the m
inputs are in general position (holds almost surely for generic θ₀ and
inputs).

**Theorem.** The number of realizable dichotomies is exactly C(m, P) of
Lemma K′.2, hence the storage threshold is sharp at m = 2P:
**κ = 2 bits per parameter, for every architecture, in the lazy
regime.**

*Proof.* In the linearized regime the trainable object is the direction
δ = θ − θ₀ ∈ ℝ^P, and the decision on x_i is sign(φ(x_i)·δ): a
homogeneous linear threshold in the P-dimensional gradient-feature
space. Apply Theorem K′ with n = P. Architecture enters only through
φ — and Cover's count depends on no property of φ beyond general
position. ∎

**Scope.** This proves the architecture-independence half of
Conjecture A for the lazy regime: depth, attention, convolutions — all
give κ = 2 when training stays linearized. The remaining gap to full K
is exactly the *feature-learning (rich) regime*; there our bench
already measures κ ≈ 2.1 on a rich-regime MLP (E2), so the constant
empirically survives feature learning — that survival is now the
precise open statement. Empirical check of K′′: E6 (two unrelated
feature architectures, same threshold at 2 bits per trainable
parameter).

---

## Theorem P′′ (full P without the compute bound, and on average)

*Hypotheses deleted from full P: (i) bounded compute, or (ii)
worst-case task choice. Each deletion yields a theorem.*

**(i) Unbounded-compute version.** For ANY task with a consistent
program of length ℓ* and per-candidate check time t, Levin-ordered
enumeration returns a consistent program of length ≤ ℓ* (hence
LCDL ≤ CDL + O(1)) after at most 2^{ℓ*+1} candidate checks.
*Proof.* Enumerate programs in nondecreasing length; there are
< 2^{ℓ*+1} programs of length ≤ ℓ*; the target is among them; the first
consistent one returned is no longer than it. ∎
So the *description* claim of P holds for every algorithmic class;
only worst-case *time* is exponential.

**(ii) Average-case version.** If tasks are drawn from any distribution
𝒟 with E_𝒟[2^{Kt(task)}] ≤ S, the expected number of candidate checks
of (i) is ≤ 2S — polynomial whenever S is.
*Proof.* Linearity of expectation over the bound in (i). ∎
Under such priors the Proposition-5 wall has exponentially small mass:
a uniformly random parity on n bits has prior mass ~2^{−n} under any
lightweight-Kt prior aligned with a compositional library.

**(iii) Sharpening: worst-case-efficient full P is FALSE under standard
cryptography.** Goldreich–Goldwasser–Micali pseudorandom functions are
computable by small circuits (small CDL) yet, by definition,
indistinguishable from random functions to every efficient observer. An
efficient learner achieving small LCDL on them from query access would
distinguish them from random (a random function admits no consistent
short responder — Theorem 8's F2 negative control). Hence, if one-way
functions exist, no efficient algorithm attains small LCDL on all
small-CDL targets (Kearns–Valiant hardness, assembled). ∎
**Consequence.** The "open" worst-case half of P is not awaiting a
prover — it is almost certainly false. The true and proven content of P
is P′ (compositional families) + P′′(ii) (simple-on-average
distributions), and the genuine open problem is *characterizing the
distributions* for which learning is efficient. That is the correctly
posed frontier.

---

## Theorem K‴ (the rich regime, resolved by splitting the object)

*Circumvention: delete the hidden assumption that "capacity" is one
number. It is three — and separating them settles two and sharpens the
third.*

**(a) Conservation bound (proven, trivial but load-bearing).** Encode
each of P weights in b bits. Stored extractable information
≤ P·b. Define the storage efficiency η = stored bits / (P·b). Then
η ≤ 1 always. ∎ (Counting.)

**(b) The representational constant "2" is FALSE (assembled).** With
unbounded weight precision, ReLU networks can memorize N labeled points
with Õ(√N) parameters (bit-extraction constructions; Vardi, Yehudai &
Shamir 2021, arXiv:2110.03187) — i.e., ~√N bits per parameter:
**no architecture-universal representational constant exists.** The
resolution of the apparent paradox with (a): those constructions use
Θ̃(√N)-bit precision per weight, so total weight-bits ≈ √N·√N = N and
η ≈ 1 — bit extraction reallocates the SAME bit budget from many
low-precision weights to few high-precision ones. Conjecture A, read
representationally, is thereby refuted; the constant 2 was never a
property of architectures. Verified numerically in E7: a single float64
parameter stores ~50 bits.

**(c) Robustness bound (proven).** If stored behavior must survive
independent relative weight perturbations of size ρ (flat minima; int8
robustness is ρ ≈ 2⁻⁸), then distinguishable robust weight
configurations number ≤ (c/ρ)^P, so stored bits ≤ P·log₂(c/ρ):
**κ ≤ log₂(c/ρ) bits/param.**
*Proof.* Robust behaviors are constant on ρ-cubes; a [−W,W]^P box
contains ≤ (cW/ρW)^P disjoint such cubes; distinguishable behaviors
inject into cubes. ∎ E7 measures exactly this line: recoverable bits
per parameter ≈ log₂(1/ρ).

**(d) What remains — the dynamical constant, now minimal and sharp.**
By (b) capacity is not representational; by (c) it is precision/
robustness-bounded; by K′/K′′ it equals 2 exactly in solvable regimes;
by E2 it degrades with training noise. The last open statement of
Conjecture A is therefore purely dynamical:
    *gradient training with a natural noise floor produces storage at
    η ≈ 2/b_eff — why 2?*
Every other reading is now settled. The designated attacks remain §5's
(Gardner–Talagrand; trajectory rate–distortion), but the target has
shrunk from "a law of architectures" to "a constant of SGD".

---

## Theorem P‴ (the crypto wall is an ACCESS phenomenon, not a
computational one)

*Circumvention: delete "query-only access" — the hypothesis that
Kearns–Valiant hardness actually uses.*

**(a) Genesis/white-box version (proven).** Let the target f be
produced by any efficient process: f = Train(code, data, seed), with
total genesis description g = |code| + |data-reference| + |seed| bits.
Then LCDL(f) ≤ g + O(1), achieved by the efficient "learner" that
simply carries the genesis and replays it.
*Proof.* The genesis triple IS a program computing f; replaying it is
efficient by assumption. ∎
**Under white-box/genesis access, worst-case efficient P is TRUE for
every efficiently-created target.** In particular every trained model's
shortest known description is its recipe — verified bit-for-bit in E8.

**(b) Query-access version stays FALSE under OWF** (P′′ iii): GGM
pseudorandom functions have tiny genesis (a seed!) yet defeat every
efficient *query* learner. Note what this juxtaposition proves: the
same object is trivially compressible with the seed and cryptographically
incompressible without it. The obstruction was never computation — it
is *what you are given*.

**(c) Corollary (which wall binds whom).** Open-weight models on disk
are white-box, genesis-published artifacts: Kearns–Valiant never binds
them. The operative limits for capability-in-12-GB are Theorem K (weight
storage), Lemma 2 (fact floors) and Theorem 8/F2 (structure under Q) —
never the crypto wall. The crypto wall binds exactly one actor: a
learner facing an adversarial black box.

**(d) The true remaining open object.** Between C = poly (black-box:
hard) and C = C_train (genesis: trivial) lies the time–description
tradeoff curve LCDL(C). Distillation, quantization and this repo's
whole toolchain are empirical points in its interior. Characterizing
the curve is the correctly-posed remainder of P — and it subsumes the
Densing law (Conjecture B) as its time evolution.

---

## The ledger (updated)

| Statement | Status |
|---|---|
| K′: κ = 2 bits/param for the perceptron, sharp | **PROVEN** (Cover 1965; full proof above; E4) |
| K′′: κ = 2 for EVERY architecture in the lazy regime | **PROVEN** (above; E6) |
| K‴a: conservation η ≤ 1 | **PROVEN** (counting) |
| K‴b: representational universal constant | **FALSE** (bit extraction; E7) — Conjecture A amended |
| K‴c: robust capacity κ ≤ log₂(c/ρ) | **PROVEN** (packing; E7 measures the line) |
| K‴d: the dynamical constant — why SGD sits at η ≈ 2/b_eff | OPEN — the minimal remaining core of Conjecture A |
| P′: exponential programs-vs-weights separation; exponent-reducing libraries | **PROVEN** (above; E5) |
| P′′(i): LCDL = CDL + O(1) for all classes, unbounded compute | **PROVEN** (Levin bound) |
| P′′(ii): efficient on average over simple-on-average task distributions | **PROVEN** (expectation bound) |
| P‴a: worst-case efficient P under white-box/genesis access | **PROVEN** (genesis replay; E8) |
| P, worst-case efficient, query-only access | **FALSE under one-way functions** (P′′ iii) — binds black boxes only |
| P‴d: the tradeoff curve LCDL(C) | OPEN — the correctly-posed remainder (subsumes Densing law) |
| Theorem 8: ε-indistinguishable pseudo-models at fixed budget | **PROVEN** (INDISTINGUISHABILITY.md; E3) |
