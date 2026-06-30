# Paper outline — "Fractional-order memory as a parameter-efficient inductive bias"

A realistic skeleton for a submission. **Target venue, honestly ranked:**
1. **NeurIPS / ICML / ICLR** (most realistic for this result).
2. **Nature Machine Intelligence** (possible *iff* the real-data results are
   strong and the framing is broad; not *Nature* proper, which essentially never
   takes a sequence-model method paper).

The claim is deliberately narrow and provable, not the "wrong algebra" thesis —
which our own experiments (`RESULTS.md`) refute via the 3D-vs-4D dimensional wall.

## One-sentence contribution

A single learnable fractional order `q` realizes power-law long memory that a
diagonal state-space model needs `Θ(log L)` modes to approximate, giving a
horizon-independent, parameter-efficient inductive bias for polynomial long-range
dependence — with a clearly characterized regime where it does *not* help.

## Section plan

1. **Introduction.** Long-range dependence in sequence models; the parameter
   cost of memory; fractional calculus (Atangana–Baleanu kernels) as a
   horizon-free parametrization. State the bounded claim up front.
2. **Background.** Grünwald–Letnikov / fractional integration; diagonal SSMs
   (S4/S4D, HiPPO); sum-of-exponentials approximation of power laws.
3. **Theory.** Proposition 1 + Theorem 2 + Corollary from `THEORY.md`. Full
   proof of the `Ω(log L)` lower bound (tighten the sketch).
4. **A fractional memory layer.** Learnable `q` (and multi-order mixtures);
   drop-in channel; cost analysis. Where it composes with nonlinearities.
5. **Experiments.**
   - Synthetic memory regimes (power vs exp) — the cross-over
     (`benchmark_longmemory.py`). Establishes the inductive bias and its limit.
   - **Long Range Arena** at equal parameter budget vs S4/S4D, Mamba,
     Transformer. *This is the gate.* (TODO — requires a real training stack.)
   - A real long-memory time series (traffic / climate / ECG). (TODO.)
   - Ablations: fixed vs learned `q`; single vs mixture of orders; horizon
     scaling confirming the `log L` prediction.
6. **Limitations.** Regime-specificity (exp memory); single-channel theory vs
   full model; modest constant-factor/`log` gap; no claim beyond memory.
7. **Conclusion.** A cheap, principled memory primitive — an added tool.

## Honest gating checklist (do these IN ORDER; stop if one fails)

- [x] Cross-over on synthetic regimes (done; power-favorable, exp-unfavorable).
- [x] Theorem with stated boundary (done; `THEORY.md`).
- [ ] **Beat-or-match S4/S4D at equal params on Long Range Arena.** If this
      fails, there is no ML paper — only a (smaller) applied-math note on
      power-law-kernel approximation. Reassess here.
- [ ] One real-world long-memory dataset.
- [ ] Horizon-scaling experiment matching the `log L` prediction.

## What is already in this repo to build on

- `operators.py` — fractional derivative / Atangana memory operators.
- `benchmark_longmemory.py` — the regime cross-over experiment + learnable-`q`
  fractional layer and a diagonal-SSM baseline (numpy reference implementations).
- `THEORY.md` — the proposition/theorem/corollary and the honest boundary.
- `RESULTS.md` — the negative results that scope the project (Trinition / the
  "wrong algebra" thesis is dimensionally precluded; fractional memory is the
  surviving signal).

## Next engineering step

Port the fractional layer + SSM baseline to a GPU stack (PyTorch) and run Long
Range Arena. The numpy versions here are the spec; they define exactly what to
reproduce. Everything stays falsifiable: the gating checklist says where to quit.
