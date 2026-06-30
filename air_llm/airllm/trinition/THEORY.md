# A parameter-efficiency separation for fractional memory

This note states the theorem the empirical study points to, with a proof sketch.
It is the "citable" core: it explains *why* a single learnable fractional order
can match a many-parameter state-space model on long-memory data — and why that
advantage is bounded to a specific memory regime, not universal.

It is deliberately modest and honest. It does **not** prove anything about
language, attention, or "destroying the industry"; it proves a clean fact about
the parameter cost of representing power-law memory.

## Setup

A causal linear memory operator maps an input stream `(x_t)` to
`y_t = Σ_{k≥0} h_k x_{t−k}` via a kernel `h`. We compare two parametrizations.

- **Fractional operator** of order `q ∈ (0,1)`: the kernel is the
  Grünwald–Letnikov / fractional-integration kernel
  `h^frac_k(q) = binom(k+q−1, k) ~ k^{q−1} / Γ(q)` as `k → ∞`.
  It is described by the **single scalar `q`** (plus an O(1) readout).

- **Diagonal state-space model (SSM)** with `n` modes (the S4D core):
  `h^ssm_k = Σ_{i=1}^{n} c_i a_i^{k}`, i.e. a sum of `n` geometric/exponential
  sequences. It has `Θ(n)` parameters (`a_i, b_i, c_i`).

We measure cost by the number of parameters needed to realize a target kernel
over a horizon `L` (lags `0 ≤ k ≤ L`) to relative accuracy `ε`.

## Proposition 1 (fractional side, exact, O(1))

For every `q ∈ (0,1)`, the power-law memory kernel `k^{q−1}` is realized
*exactly* (up to the constant `1/Γ(q)`) by the fractional operator of order `q`,
i.e. by **one** parameter, at every horizon `L`.

*Proof.* Immediate from the definition of `h^frac_k(q)` and the asymptotic
`binom(k+q−1,k) = Γ(k+q)/(Γ(q)Γ(k+1)) ~ k^{q−1}/Γ(q)` (Stirling). The order `q`
sets the decay exponent directly; no dependence on `L`. ∎

## Theorem 2 (SSM side, lower bound Ω(log L))

Fix `q ∈ (0,1)`. To approximate the power-law kernel `g(k) = k^{q−1}` on
`1 ≤ k ≤ L` by a sum of `n` exponentials `Σ_i c_i a_i^k` with uniform relative
error `ε`, one needs

```
    n = Ω( log L · log(1/ε) )           (and this rate is achievable).
```

Hence a diagonal SSM matching power-law memory over horizon `L` requires
`Θ(log L · log(1/ε))` modes, i.e. `Θ(log L · log(1/ε))` parameters.

*Proof sketch.* `g` is completely monotone, so the question is the optimal
sum-of-exponentials (equivalently, rational-in-the-Laplace-domain) approximation
of `x^{q−1}` on a geometric range `[1, L]`. The number of exponential nodes
needed for a completely monotone power kernel on a range spanning `R` decades
grows like `log R` per fixed accuracy, with a `log(1/ε)` factor for the accuracy
— this is the classical sum-of-exponentials / quadrature-of-`x^{−s}` result
(Beylkin & Monzón, *Approximation by exponential sums revisited*, ACHA 2010; and
the `1/x^q` quadrature of Beylkin–Monzón / Braess–Hackbusch). A matching lower
bound follows because each exponential resolves a bounded multiplicative band of
lags `[a^{−1}_i scale]`, and covering `L` lags with geometrically-spaced bands
requires `Ω(log L)` of them; pushing relative error to `ε` inside each band costs
the `log(1/ε)` factor. ∎ (full version: tighten the lower bound via a
Chebyshev/Markov argument on the Laplace transform.)

## Corollary (separation)

On data whose memory kernel is power-law over horizon `L`,

```
    fractional params = O(1)        vs.        SSM params = Θ(log L · log(1/ε)).
```

The fractional operator sits strictly inside the SSM's Pareto frontier in the
long-memory regime, and the gap **grows with the horizon** `L`. This is exactly
the empirical picture in `benchmark_longmemory.py`: a 1-parameter learned order
matches a multi-mode SSM on the `power` regime.

## What the theorem does NOT say (the honest boundary)

- **It is regime-specific.** If the true memory is a *single* exponential (or a
  small fixed number), an SSM realizes it with `O(1)` parameters and the
  fractional operator is *worse* — it cannot represent a pure exponential with a
  power-law kernel. The empirical `exp` regime confirms this; the separation
  reverses. There is no free lunch (Wolpert): the win is an inductive bias for
  *polynomial* long memory, nothing more.
- **It is about a single linear memory channel**, not a full sequence model.
  A real architecture interleaves such channels with nonlinearities and mixing;
  the per-channel separation is necessary, not sufficient, for an end-to-end win.
- **`Θ(log L)` is a modest gap**, not exponential. The contribution is "a
  cheaper, horizon-independent inductive bias for power-law memory," not a
  revolution.

## Why this is the publishable core

It connects a real, classical piece of Atangana's domain (fractional calculus /
power-law memory kernels) to a precise, falsifiable statement about modern
sequence models (SSMs), with (i) a proof, (ii) a stated boundary where it fails,
and (iii) matching experiments including the failure case. That triad — claim,
limit, evidence-both-ways — is what referees reward. The grand "wrong algebra"
thesis has none of it; this does.
