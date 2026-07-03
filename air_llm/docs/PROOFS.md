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

## The ledger

| Statement | Status |
|---|---|
| K′: κ = 2 bits/param for the perceptron, sharp | **PROVEN** (Cover 1965; full proof above) |
| K: same constant for all architectures | OPEN (universality; route in §5) |
| P′: exponential program-vs-weights separation for compositional families; exponent-reducing libraries | **PROVEN** (above) |
| P: polylog LCDL for general algorithmic classes | OPEN (MCSP wall; route in §5) |
| Theorem 8: ε-indistinguishable pseudo-models at fixed budget | **PROVEN** (INDISTINGUISHABILITY.md) |
