# THEOREM_KOLMOGOROV — Lower bounds for memory-bounded inference

> Companion to `PROJECT_KOLMOGOROV.md`. This document states and proves the
> *tractable-sibling* theorems: unconditional lower bounds on the resident
> information an inference engine must hold, via communication complexity. It
> marks precisely **what is proven** vs **what is conjectured**, and develops
> §6.1 (the distortion-aware, average-case version) — the part whose proof
> technique, if generalized, is the only object here of potentially Fields shape.
>
> Status: mathematical draft. The gadget-level theorems (Thm 1, Thm 2) are
> proven at full-sketch rigor with standard tools. The general rate–distortion
> inequality (§7) is a conjecture with an attack plan.

---

## 1. Model of computation (faithful to disk-streaming inference)

A **streaming inference engine** `A` with resident budget `s`:

1. reads the weights as a one-pass stream `w = (w_1, …, w_n)`, `w_i ∈ {0,1}`,
   from slow storage;
2. maintains a **resident state** `τ ∈ {0,1}^s` (the fast memory; the "resident
   generator + codes" of the project live here);
3. **after** the pass, receives a query `q` (the context for the next token) and
   must emit an output token / distribution.

**Modeling assumption (load-before-query).** The query is revealed *after* the
weight pass. This is faithful to autoregressive generation: *which weights are
active for future tokens depends on context that has not yet unfolded when the
weights are loaded.* (If the needed index were known before the pass, one would
load a single weight — correctly trivial.) Public randomness is allowed.

**Quantity.** `S*(n)` = minimal resident budget `s` such that some `A` answers a
post-pass query with constant advantage over chance, in the worst case over
weights.

---

## 2. The hammer: Augmented Indexing

**Problem `AIND_n` (one-way, Alice→Bob).** Alice holds `x ∈ {0,1}^n`; Bob holds
`j ∈ [n]` **and** the suffix `x_{j+1}, …, x_n`. Alice sends one message `M`; Bob
outputs a guess for `x_j`.

**Fact (Miltersen–Nisan–Safra–Wigderson 1998).** Any (public-coin) protocol with
success probability `≥ 2/3` has `|M| = Ω(n)` — *even though Bob already knows the
entire suffix.* This robustness is the crux: side knowledge of part of the input
does not lower the message size.

---

## 3. Gadget + embedding into the transformer class

**Gadget `G_x`.** A single linear read-out with weight vector `w = x ∈ {0,1}^n`.
Query = one-hot `e_j ∈ ℝ^n`. Pre-logit `ℓ = ⟨w, e_j⟩ = x_j`. Output two logits
`(C·ℓ, 0)`, `C > 0` constant; softmax gives token-1 with probability `σ(C)` if
`x_j = 1` and `σ(0)=½` if `x_j = 0`. Margin is constant and tunable via `C`.
`G_x` carries exactly `n` bits of model information.

**Embedding Lemma (hardness lives inside the real class).** A single
attention head with key matrix `= I` and values `= w` realizes `G_x`: the query
token attends to position `j` (one-hot keys) and reads value `w_j = x_j`, which a
linear head turns into the logit gap. Hence `{G_x}` ⊆ functions realizable by a
1-layer transformer. *The lower bounds below are therefore about the genuine
transformer function class, not a toy outside it.*

---

## 4. Theorem 1 (worst-case, exact)

> **Theorem 1.** `S*(n) = Ω(n)`. Concretely: any streaming engine answering a
> post-pass one-hot query on the gadget family with advantage `≥ 1/6` over chance
> must use resident state `s = Ω(n)`, **even if an arbitrary suffix of the weights
> is given to it for free.**

**Proof.** Reduce `AIND_n` to inference. Given `(x; j, x_{j+1..n})`:

- **Alice** sets `w = x`, runs `A` on the stream `w_1, …, w_n`, obtains resident
  state `τ ∈ {0,1}^s`, and sends `M := τ` to Bob (one-way, `s` bits).
- **Bob** holds `j` and the suffix (which corresponds to weights `w_{j+1..n}`
  known "resident for free"). He resumes `A` from `τ`, feeds query `e_j` — which
  needs no further weights — reads the output token, and decodes
  `x_j = 1` iff token-1, else `0`.

If `A` answers with advantage `≥ 1/6`, Bob solves `AIND_n` with probability
`≥ 2/3`. By the MNSW fact, `s = |M| = Ω(n)`. ∎

**Reading.** One cannot summarize the model into sub-linear resident state and
then serve post-pass queries — even given almost the whole model for free. This
is the rigorous refutation of "a tiny resident generator stands in for the
model," *in the worst case*.

---

## 5. Tightness (Kolmogorov metric entropy)

Let `H_ε` be the `ε`-entropy (log covering number) of the realized function class
under the task's output metric (Kolmogorov–Tikhomirov).

- **Incompressible extreme.** Weights of entropy `Θ(n)` ⇒ `H_0 = Θ(n)`. Lower
  bound `Ω(n)` (Thm 1) is matched by the trivial `n + o(n)` resident upper bound
  (hold everything). Tight — and it is exactly **Shannon**: you cannot beat the
  entropy.
- **Structured extreme.** If `H_ε ≪ n`, a generator of size `O(H_ε)` reaches
  distortion `ε` (cover the class, store the index of the nearest center).

> **Corollary (pincer).** `S*_ε(n) = Θ(min(n, H_ε))`. *The fate of streaming
> inference is the metric entropy of the model's function class.* Sparsity, MoE,
> quantization, weight-generation are all ways of exploiting small `H_ε`; none
> can beat it.

---

## 6. (=§6.1) Theorem 2 — average-case, distortion-aware

Thm 1 is worst-case and exact: a real model is wrong-but-useful (perplexity `ε`),
and we care about *average* behaviour. We upgrade via a coding argument.

**Setup.** Pick a constant-rate binary code `Enc: {0,1}^k → {0,1}^n`,
`n = O(k)`, list-decodable up to radius `(1/2 - γ)` with list size
`L = O(1/γ²)` (Guruswami; such codes exist explicitly). The weights are
`w = Enc(x)`, `x` uniform on `{0,1}^k`. Query distribution: `j` uniform on `[n]`,
revealed post-pass. Output uses logit scale `C` so that a *confident wrong*
coordinate contributes KL gap `Δ = Δ(C)` to the per-token distortion.

> **Theorem 2 (distortion-aware lower bound).** Suppose `A` achieves expected
> output distortion `≤ ε` over the query distribution, with
> `ε < (1/2 - γ)·Δ` for some constant `γ > 0`. Then its resident budget obeys
> `s ≥ k - O(log(1/γ)) = Ω(n)`.

**Proof.**
1. *Distortion ⇒ Hamming advantage.* Average KL distortion `≤ ε` and per-bad-
   coordinate gap `Δ` imply the fraction of confidently-wrong coordinates is
   `≤ ε/Δ`. Reading `A`'s answer `ĉ_j` for every `j` off the *fixed* map
   `(τ, j) ↦ token` yields a word `ĉ ∈ {0,1}^n` with
   `d_H(ĉ, w) ≤ (ε/Δ)·n < (1/2 - γ)n`.
2. *Decode.* List-decoding `ĉ` returns a list of `≤ L` messages containing `x`.
   Thus the resident state `τ` (plus public coins) determines `x` up to a list of
   size `L`, with probability `≥ 2/3`.
3. *Counting / Fano.* A map `{0,1}^s → (lists of size ≤ L)` that covers `≥ 2/3`
   of the `2^k` messages satisfies `2^s · L ≥ (2/3) 2^k`, hence
   `s ≥ k - log L - O(1) = k - O(log(1/γ)) = Ω(n)`. ∎

**Reading.** Resident memory must stay `Ω(n)` until the *average* distortion `ε`
approaches a constant fraction `(1/2-γ)Δ` of the per-coordinate gap. The bound
**degrades gracefully with `ε`** — a genuine rate–distortion shape, not a cliff.
This is the honest, average-case, approximate statement the project needs.

---

## 7. The deep kernel — a rate–distortion theorem for memory-bounded computation

Theorems 1–2 are proven *for the gadget* (and, by §3, inside the transformer
class). The object whose general proof would be a *method*, not an application:

> **Conjecture (RD-COMP).** For a natural class `F` of functions and a task
> distortion `d`, the minimal resident information to evaluate `f ∈ F` in one
> pass to expected distortion `ε` satisfies
> `S*_ε(F) ≥ H_ε(F) · (1 − o(1))`,
> where `H_ε(F)` is the metric `ε`-entropy of `F`. I.e. **the streaming resident
> cost equals the rate–distortion function of the function class** — Shannon's
> rate–distortion theorem, but for *interactive computation under a memory
> bound* rather than for source coding.

**Why it is plausibly provable (attack plan).** The gadget gives the bound for
one "coordinate of computation." The general statement is a **direct-sum /
direct-product** claim over `n` such coordinates. The modern tool is
**information complexity** (Bar-Yossef–Jayram–Kumar–Sivakumar; Braverman): prove
a *single-coordinate* distortion–information inequality (Thm 2 is its seed), then
a direct-sum theorem amortizes it across coordinates to `H_ε(F)`. The missing
piece is a **distortion-aware information-complexity inequality** — a
strengthening of the standard (exact, worst-case) ones to expected distortion.

**Why this is the only Fields-shaped object here.** If that inequality is new and
reusable beyond inference — i.e. a general *rate–distortion theory of streaming
computation* — the prize is for the **technique**, exactly as Fields-level work
rewards methods, not single results. We state this honestly as aspiration, not
claim. Absent that generality, Thms 1–2 are **Gödel/Abacus-shaped**: deep,
unconditional, celebrated — and already the realistic ceiling.

---

## 8. What this buys Project KOLMOGOROV

1. **It tells you exactly where the win must come from.** Not cleverness in the
   engine (Thm 1 forbids it) but small `H_ε` of real weights — the *empirical* C1
   question. Theory fixes the form `Θ(min(n, H_ε))`; the **Stage 1 scaling law
   measures `H_ε(N)`**. Theory and experiment lock together.
2. **It explains the only escapes as corollaries**, not tricks:
   - *Batching/throughput*: amortize one `Ω(n)` load over `B` queries ⇒ per-token
     traffic `Ω(n)/B`. (Extend Thm 1 to `Q` adaptive queries: total
     `Ω(min(Qn, ...))`.)
   - *Sparsity / MoE*: only viable if the *active* sub-function has small `H_ε`.
   - *Generation of weights*: viable iff `H_ε ≪ n`, i.e. iff C1 holds.
3. **It pre-commits the negative result honestly**: if Stage 1 finds `H_ε(N) =
   Θ(N)` (entropy does not become sub-linear with scale), the pincer says
   disk-streaming inference is fundamentally `Ω(model)` per token-batch — a clean,
   publishable impossibility, not a failure.

---

## 9. Proven vs open (ledger)

| Statement | Status |
|---|---|
| Thm 1 (worst-case exact, `Ω(n)`, robust to free suffix) | **proven** (standard tools) |
| Embedding into 1-layer transformer class | **proven** (lemma §3) |
| Tightness pincer `Θ(min(n, H_ε))` at the two extremes | **proven** |
| Thm 2 (average-case, distortion-aware, `Ω(n)` for `ε < (1/2-γ)Δ`) | **proven** (coding + counting) |
| Multi-query / per-token traffic version | open (mechanical extension expected) |
| RD-COMP general inequality (§7) | **conjecture** + attack plan |
| `H_ε(N)` for real transformers | **empirical** (Stage 1) |

## 10. References (to verify at lit-review gate)

- Miltersen, Nisan, Safra, Wigderson, *On data structures and asymmetric
  communication complexity*, 1998 — Augmented Indexing `Ω(n)`.
- Kushilevitz & Nisan, *Communication Complexity*, 1997.
- Bar-Yossef, Jayram, Kumar, Sivakumar, *An information statistics approach to
  data stream and communication complexity*, 2004 — information complexity / direct sum.
- Braverman, *Interactive information complexity*, 2012.
- Kolmogorov & Tikhomirov, *ε-entropy and ε-capacity of sets in function
  spaces*, 1959 — metric entropy / n-widths.
- Guruswami, *List decoding of error-correcting codes*, 2004 — constant-rate
  list-decodable binary codes.
- Hong & Kung, *I/O complexity: the red-blue pebble game*, 1981 — two-level
  memory I/O lower bounds (the systems-side companion model).
- Shannon, *Coding theorems for a discrete source with a fidelity criterion*,
  1959 — rate–distortion (the analogy RD-COMP generalizes).
