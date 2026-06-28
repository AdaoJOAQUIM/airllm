# PROOFS_KOLMOGOROV — Complete proofs and the honest open crux

> Self-contained, gap-free proofs of the results that **are** provable (Gödel/
> Abacus-grade), plus the rigorous *proof program* for the Fields-shaped object
> with its single open lemma isolated.
>
> **Intellectual-honesty clause.** RD-COMP (§4) is an **open** research
> conjecture. No verified proof of it is given here, and none should be claimed:
> a confident-looking fake proof of an open problem is the worst possible
> outcome. What is delivered is (i) two theorems proven in full, standing on
> cited prior theorems, and (ii) a reduction of RD-COMP to one clean, named,
> unproven lemma, with the strongest honest partial progress. A proof is only
> "real" once checked by the community; treat §3–§4 as a referee-ready draft, not
> a certificate.

Notation: `[n] = {1,…,n}`, `H(·)` binary entropy, `log = log₂`, `ln` natural log.

---

## 0. The computational model, precisely

A **randomized one-pass streaming inference engine** with resident budget `s` is
a tuple `A = (R, σ₀, U, O)` where `R` is a public random string, `σ₀ ∈ {0,1}^s`
the initial state, `U : {0,1}^s × {0,1} → {0,1}^s` the per-symbol update, and
`O : {0,1}^s × 𝒬 → 𝒴` the output map on query space `𝒬`. On weight stream
`w ∈ {0,1}^n` it computes `τ(w) = U(⋯U(U(σ₀,w₁),w₂)⋯,w_n) ∈ {0,1}^s`, then on a
post-pass query `q` returns `O(τ(w), q)`. All maps may depend on `R`.

This is exactly disk-streaming inference: weights flow once from slow storage,
`τ` is the bounded resident state ("generator + codes"), the query (context for
the next token) arrives after the pass.

---

## 1. External theorems used (black boxes)

We stand on three established results; our proofs are unconditional **given**
these. (Verify citations at the lit-review gate.)

- **(AIND)** *Augmented Indexing lower bound* (Miltersen–Nisan–Safra–Wigderson
  1998). In `AIND_n`: Alice has `x∈{0,1}^n`; Bob has `j∈[n]` and `x_{j+1..n}`;
  one message Alice→Bob; Bob outputs `x_j`. The one-way public-coin randomized
  communication complexity with error `≤ 1/3` is `R^{→}_{1/3}(AIND_n) = Ω(n)`.
- **(LIST)** *List-decodable codes* (Guruswami; random-coding existence). For
  every constant `γ∈(0,1/2)` there is a binary code `Enc:{0,1}^k→{0,1}^n` of
  constant rate `ρ = k/n = Ω(γ²)`, decodable from a `(1/2−γ)`-fraction of errors
  to a list of size `L ≤ L₀(γ) = O(γ^{-2})`.
- **(KT)** *Metric entropy* (Kolmogorov–Tikhomirov 1959). For a function class
  `𝓕` with output pseudometric `d`, `H_ε(𝓕) = log 𝒩(𝓕,d,ε)` (log covering
  number) is the information-theoretically minimal description length to specify
  an element of `𝓕` up to output distortion `ε`.

---

## 2. The gadget and its realizability

**Gadget `G_x`** (`x∈{0,1}^n`): on one-hot query `e_j∈ℝ^n`, the correct next
token is `t_j := x_j ∈ {0,1}`. (Realized by pre-logit `⟨x,e_j⟩ = x_j` feeding a
2-token head; logit scale chosen in §3.2.) `G_x` holds exactly `n` model bits.

**Lemma 2.1 (embedding).** Each `G_x` is computed by a 1-layer, 1-head attention
network. *Proof.* Keys `K=I_n`, so query token `e_j` attends to position `j`;
values `V=x` return `x_j`; a linear+softmax head emits the 2-token logits. ∎

Consequence: the hardness below is **inside** the transformer function class, not
a synthetic object beside it.

---

## 3. The proven theorems

### 3.1 Theorem 1 (worst-case, exact)

> **Theorem 1.** Let `A` be any randomized one-pass engine that, on the family
> `{G_x : x∈{0,1}^n}`, outputs a guess `g` of `x_j` for a post-pass one-hot query
> `e_j` with `Pr[g = x_j] ≥ 2/3` (probability over `R`), **even when the suffix
> `x_{j+1..n}` is provided to `O` for free**. Then `s = Ω(n)`.

**Proof.** Reduce `AIND_n` to using `A`. Given an instance `(x; j, x_{j+1..n})`
with shared public coins `R`:
1. **Alice** sets `w := x`, computes `τ(w) ∈ {0,1}^s`, sends `M := τ(w)`.
2. **Bob** receives `M = τ`, holds `j` and `x_{j+1..n}`, and outputs
   `g := O(τ, e_j)` (the free suffix is exactly the side information `O` is
   permitted). By hypothesis `Pr[g=x_j] ≥ 2/3`.

This is a one-way protocol for `AIND_n` with error `≤ 1/3` and message length
`|M| = s`. By **(AIND)**, `s ≥ R^{→}_{1/3}(AIND_n) = Ω(n)`. ∎

*Remark (why "guess with advantage" is the right inference notion).* Any engine
achieving cross-entropy `< log2` on the gadget corpus predicts `x_j` better than
chance; Theorem 2 makes this quantitative with the actual perplexity metric.

### 3.2 Theorem 2 (average-case, distortion = perplexity) — this is §6.1

We now use the **true scoring metric** of language models: log-loss /
cross-entropy (perplexity `= e^{avg loss}`), which crucially **penalizes
hedging** and so forces commitment.

Setup: fix `γ∈(0,1/2)`, take the code `Enc` from **(LIST)** with parameters
`(k,n,L₀(γ))`, `k = ρn`. Weights are `w = Enc(x)`, `x∈{0,1}^k`. Query `j` uniform
on `[n]`; correct token `t_j = w_j` (the codeword bit). The engine outputs a
distribution `p̂_j` on `{0,1}`; its per-query loss is `ℓ_j = −ln p̂_j(t_j)`; its
**average distortion** is `ε := 𝔼_{j∼[n]} ℓ_j` (natural-log perplexity `e^{ε}`).

> **Theorem 2.** Fix `λ ∈ (0, ln2)` and let `γ` be the code parameter. If `A`
> achieves average distortion `ε ≤ λ(1/2 − γ)` on every `w=Enc(x)`, then
> `s ≥ k − log L₀(γ) − O(1) = Ω(n)`.

**Proof.**
*(a) Distortion ⇒ Hamming closeness.* Fix `x` and coins `R`. Read the engine's
hard decision `ĉ_j := argmax_b p̂_j(b)` for all `j∈[n]` off the fixed map
`(τ(w), e_j) ↦ p̂_j`. Call `j` *good* if `ℓ_j < λ`. For good `j`,
`p̂_j(t_j) = e^{−ℓ_j} > e^{−λ} > e^{−ln2} = 1/2`, hence `ĉ_j = t_j = w_j`. By
Markov on `ℓ_j ≥ 0` with mean `ε`, the fraction of non-good `j` is
`≤ ε/λ ≤ (1/2 − γ)`. Therefore
`d_H(ĉ, w) ≤ (ε/λ)·n ≤ (1/2 − γ)·n`.

*(b) Decode.* By **(LIST)**, list-decoding `ĉ` yields a list `𝓛(τ,R)` of `≤ L₀`
messages containing `x` (since `w=Enc(x)` is within radius `(1/2−γ)` of `ĉ`).
Thus, for every `x`, the resident state `τ` (with coins `R`) determines `x` up to
a list of size `≤ L₀`.

*(c) Counting.* Fix any coins `R`. The composed map `Φ_R : {0,1}^s → (subsets of
{0,1}^k of size ≤ L₀)`, `τ ↦ 𝓛(τ,R)`, must satisfy `x ∈ Φ_R(τ(Enc(x)))` for all
`2^k` values of `x`. Each state contributes at most `L₀` messages, so the number
of covered messages is `≤ 2^s · L₀`. Hence `2^s · L₀ ≥ 2^k`, giving
`s ≥ k − log L₀(γ) = Ω(n)`. ∎

*Sharpness of shape.* The hypothesis `ε ≤ λ(1/2−γ)` makes the bound **degrade
continuously** as the tolerated perplexity grows: this is a genuine
rate–distortion curve, not a worst-case cliff. The metric is exactly the one LMs
are trained and evaluated on.

*Average-over-`x` variant.* If the `ε`-guarantee holds only in expectation over
`x` too, apply Markov to keep the `2/3` of `x` with per-instance distortion
`≤ 3ε`; counting then gives `s ≥ k − log L₀ − log(3/2) = Ω(n)`.

### 3.3 Corollary (the pincer) — tightness via (KT)

> **Corollary 3.** With `S*_ε` the minimal resident budget for average distortion
> `ε`: `S*_ε(𝓕) = Θ(min(n, H_ε(𝓕)))` on the realized class.

*Lower bound.* Theorems 1–2 give `Ω(n)` when the family has `H_ε = Θ(n)`
(incompressible codes). *Upper bound.* For any class, an `ε`-cover of size
`2^{H_ε}` (by **(KT)**) lets the engine store the index of the nearest center
resident, achieving distortion `ε` with `H_ε + o(H_ε)` bits; and `n+o(n)` always
suffices. ∎

This is the operational heart: **the resident cost of streaming inference is the
metric entropy of the model's function class.** Sparsity / MoE / quantization /
weight-generation are mechanisms to make `H_ε ≪ n`; none can beat it.

---

## 4. Theorem 4 (RD-COMP, now proven) — resident cost = metric entropy

Previously stated as a conjecture routed through a hard direct-sum lemma
(DS-dist). That route was **unnecessarily hard**. A classical *rate–distortion
converse* (sphere-packing / Fano-style) proves the general statement directly,
and — crucially — **makes coordinate correlation irrelevant**, because it never
decomposes the function into coordinates at all. We therefore promote it from
conjecture to theorem in the clean model below, and recalibrate its standing
honestly in §4.3.

### 4.1 Distortion as a metric

Let `δ` be a **metric** on the per-query output space — e.g. total variation on
next-token distributions, or `ℓ₂` on logits. Induce the pseudometric on the
function class `𝓕`:
`d(f, g) := 𝔼_{q∼Q} δ(f(q), g(q))`,
which is symmetric and obeys the triangle inequality (linearity of expectation +
triangle inequality of `δ`). Let `N(𝓕,d,ε)`, `M(𝓕,d,ε)` be the `ε`-covering and
`ε`-packing numbers; `H_ε := log N(𝓕,d,ε)` is the metric entropy (KT). Standard
fact: `N(2ε) ≤ M(2ε) ≤ N(ε)`.

**Resource accounting.** `s` = bits of **persistent post-pass resident state**
`τ(w)` (the fast memory that must remain to answer queries). Transient stream-time
scratch is not counted (faithful to "what the resident generator must hold");
this is stated as a modeling choice, not hidden.

The engine's distortion on the true `f` (weights `w_f`) is
`D(f) := 𝔼_R d(f, h_{τ(w_f),R})`, where `h_{τ,R}(q)=O(τ,q;R)` is its post-pass
output function.

### 4.2 The theorem and its proof

> **Theorem 4.** In the model of §4.1, the minimal persistent resident budget
> achieving worst-case distortion `D(f) ≤ ε/2` for all `f ∈ 𝓕` satisfies
> `H_{2ε}(𝓕) − 1 ≤ S*_{ε}(𝓕) ≤ H_ε(𝓕) + O(1)`.
> Hence, up to the standard `ε`-doubling of metric entropy,
> **`S*_ε(𝓕) = Θ(H_ε(𝓕))`** — the resident cost of one-pass inference *equals the
> metric entropy (rate–distortion function) of the model's function class.**

**Proof — lower bound (packing converse).**
Let `{f_1,…,f_M}` be a maximal `2ε`-packing, `M = M(𝓕,d,2ε)`, so
`d(f_i,f_j) > 2ε` for `i≠j`. By hypothesis `D(f_i) ≤ ε/2` for all `i`, so
`𝔼_R[ (1/M) Σ_i d(f_i, h_{τ_i,R}) ] = (1/M) Σ_i D(f_i) ≤ ε/2`.
Fix coins `R*` attaining at most the mean. By Markov, at most `M/2` indices have
`d(f_i, h_{τ_i,R*}) > ε`; call the remaining `≥ M/2` indices **good**.

Suppose two good indices `i≠j` collide: `τ(w_i) = τ(w_j) = τ`. Then the post-pass
output function `H := h_{τ,R*}` is *identical* for both (it depends only on `τ`
and `R*`, not on which weights produced `τ`). Triangle inequality:
`d(f_i,f_j) ≤ d(f_i,H) + d(H,f_j) ≤ ε + ε = 2ε`, contradicting `d(f_i,f_j)>2ε`.
So `τ(·)` is **injective on the good set**, giving `2^s ≥ M/2`, i.e.
`s ≥ log M(𝓕,d,2ε) − 1 ≥ log N(𝓕,d,2ε) − 1 = H_{2ε} − 1`. 

**Proof — upper bound (covering).** Take a minimal `ε`-cover `C`,
`|C| = 2^{H_ε}`. During the pass the engine identifies the cover element nearest
to `f` (unbounded transient scratch, freed afterward) and keeps **only its index**
(`H_ε` bits) resident. Answering a query by evaluating that cover element gives
distortion `≤ ε`. So `S*_ε ≤ H_ε + O(1)`. ∎

**Perplexity corollary (Pinsker bridge).** Cross-entropy is not a metric, but TV
is, and `TV ≤ √(KL/2)` (Pinsker). An engine with average cross-entropy distortion
`≤ ε²/2` therefore has TV-distortion `≤ ε/2`, so Theorem 4 applies with the
perplexity metric at the cost of a square root in the constant.

### 4.3 Honest recalibration of the prize

Good news: **the lemma is now a theorem** — the "open crux" (correlated
coordinates) dissolves because the packing argument never decomposes into
coordinates. Sobering news, stated plainly: **the proof is the classical
rate–distortion converse** (sphere-packing + Fano + covering achievability),
transplanted to the streaming-inference model. That makes Theorem 4 **correct,
general, and clean — but not a new method.** It is a *characterization*, of
Abacus/Gödel *flavour* at most, and honestly closer to "a clean application of
1959-era information theory" than to a medal-shaped breakthrough.

The earlier framing — that a *new* distortion-aware information-complexity
inequality robust to correlation would be Fields-shaped — is now **moot for this
result**: we did not need such an inequality, so we did not invent one. If a
genuinely novel technique lives anywhere here, it is **not** in Theorem 4; it
would be in *computing* `H_ε(𝓕)` for the transformer class (an
approximation-theory problem, see §4.4), which Theorem 4 reduces the whole
question to but does **not** solve.

### 4.4 Where the remaining mathematical content actually lives

Theorem 4 reduces "how cheap can streaming/generative inference be?" to **one
quantity**: `H_ε(transformers)`. That is now an **approximation-theory** question
(metric entropy / Kolmogorov n-widths of the transformer function class), not a
complexity-theory one. It is exactly what **Stage 1 estimates empirically**
(`b*(N) ≈ H_ε(N)/n`). Proving `H_ε(N) = o(N)` (or a plateau) for real
architectures is the open problem the theorem now isolates — and it is genuinely
deep, but it is analysis/approximation theory, not communication complexity.

---

## 5. Ledger (recalibrated, honest)

| Result | Honest tier | Status |
|---|---|---|
| Thm 1 (worst-case, `Ω(n)`, free suffix) | clean reduction | **proven** given (AIND) |
| Thm 2 / §6.1 (perplexity-distortion, `Ω(n)`) | coding + counting | **proven** given (LIST) |
| **Thm 4 — `S*_ε(𝓕)=Θ(H_ε(𝓕))`** | rate–distortion converse (classical-flavour) | **proven** given (KT) |
| RD-COMP general (incl. correlated classes) | — | **proven** (= Thm 4; no DS-dist needed) |
| New distortion-aware info-complexity method | the would-be Fields object | **not produced** (turned out unnecessary) |
| `H_ε(N)` for real transformers `= o(N)?` | the real open problem | **open** (approximation theory); estimated in Stage 1 |

**Net:** the systems→math reduction is now a *proven characterization*. The
honest prize ceiling for the proof itself is Abacus/Gödel-*flavour* at best; the
only remaining frontier-grade unknown is the value of `H_ε` for transformers.

## 6. References (verify at lit-review gate)
MNSW 1998 (Augmented Indexing); Kushilevitz–Nisan 1997; Guruswami 2004 (list
decoding); Kolmogorov–Tikhomirov 1959 (metric entropy / `ε`-entropy); Shannon
1948/1959 (rate–distortion converse); Cover–Thomas, *Elements of Information
Theory* (sphere-packing converse, Fano); BJKS 2004 / Braverman 2012 (information
complexity — the harder route Theorem 4 shows is **not** needed here).
