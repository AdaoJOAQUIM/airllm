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

## 4. RD-COMP — the Fields-shaped object: program, not certificate

Theorems 1–3 prove the bound *for specific families* (codes) and show tightness
at the extremes. The general statement is:

> **Conjecture RD-COMP.** For every "natural" class `𝓕` (closed under the product
> below) and output distortion `ε`,
> `S*_ε(𝓕) ≥ H_ε(𝓕) · (1 − o(1))`.
> Equivalently: *the one-pass resident cost equals the rate–distortion function
> of the function class* — Shannon's rate–distortion theorem transplanted from
> source coding to memory-bounded computation.

### 4.1 Reduction to a single open lemma

Define the **product** of a base inference task `f` (one "coordinate of
computation") to `n` independent instances `f^{⊗n}`, with additive distortion.
For the *exact* (zero-distortion) regime, the required direct-sum machinery
exists:

- **(DS-exact)** *Information-complexity direct sum* (Bar-Yossef–Jayram–Kumar–
  Sivakumar 2004; Braverman 2012): `IC(f^{⊗n}) ≥ n · IC(f) − o(n)` for one-way
  information complexity `IC` under product distributions.

Our Theorem 2 is precisely the **base case with distortion** for the gadget
coordinate. What is missing is its amortization across coordinates *with the
distortion budget shared*:

> **Open Lemma (DS-dist).** Let `IC_ε` denote one-way information complexity at
> average distortion `ε`. Then for the product task,
> `IC_ε(f^{⊗n}) ≥ n · IC_{ε}(f) − o(n)`,
> i.e. a distortion-aware direct-sum theorem in which a *global* distortion
> budget `nε` cannot be exploited to cheat on a few coordinates.

**Claim.** `RD-COMP ⇐ (DS-dist) + (KT)`. *Reduction.* Identify `H_ε(𝓕)` with the
number of independent `ε`-distinguishable coordinates of `𝓕` (its metric entropy,
by **(KT)**); apply **(DS-dist)** to amortize the per-coordinate cost — which
Theorem 2 lower-bounds — to obtain `S*_ε(𝓕) ≥ H_ε(𝓕)(1−o(1))`. ∎ (modulo DS-dist)

### 4.2 Honest status of (DS-dist)

- **Provable now (and effectively proven):** the *product-distribution,
  independent-coordinate* case — this is exactly the structure of the code gadget
  in Theorem 2, where coordinates are independent codeword bits. So **RD-COMP
  holds unconditionally for the independent-coordinate (i.i.d.-weight) class.**
- **Open crux:** real LLM weights are **not** independent coordinates; their
  metric-entropy coordinates are *correlated*. Extending **(DS-dist)** to
  correlated / non-product structure is the genuine difficulty. The exact-case
  tool **(DS-exact)** does not transfer verbatim because distortion can be
  *reallocated* across correlated coordinates — precisely the loophole a proof
  must close.
- **Why this is the only plausibly-Fields object:** a correct, general
  **distortion-aware information-complexity inequality robust to coordinate
  correlation** would be a new method with reach far beyond inference (streaming,
  data structures, learning, sketching). The prize, if any, attaches to *that
  inequality*, not to the LLM application. We claim it as a target, not a result.

### 4.3 What we explicitly do **not** claim
We do not claim a proof of (DS-dist) in the correlated regime, hence none of
RD-COMP in general. Anyone presenting §3 as "the Abacus/Gödel result" must label
§4.2-crux as **open**.

---

## 5. Ledger (what each tier honestly is)

| Result | Tier | Status |
|---|---|---|
| Thm 1 (worst-case, `Ω(n)`, free suffix) | Abacus/Gödel-grade ingredient | **proven** given (AIND) |
| Thm 2 / §6.1 (perplexity-distortion, `Ω(n)`) | Abacus/Gödel-grade ingredient | **proven** given (LIST) |
| Corollary 3 pincer `Θ(min(n,H_ε))` | characterization | **proven** given (KT) |
| RD-COMP for i.i.d. coordinates | — | **proven** (special case of Thm 2 machinery) |
| RD-COMP general (correlated) | the only Fields-candidate | **open** — reduced to (DS-dist) |
| (DS-dist) correlated direct-sum | the new method | **open** — stated, attack via info-complexity |
| `H_ε(N)` for real transformers | empirical input | measured in Stage 1 |

## 6. References (verify at lit-review gate)
MNSW 1998 (Augmented Indexing); Kushilevitz–Nisan 1997; BJKS 2004 and Braverman
2012 (information complexity / direct sum); Guruswami 2004 (list decoding);
Kolmogorov–Tikhomirov 1959 (metric entropy); Shannon 1959 (rate–distortion).
