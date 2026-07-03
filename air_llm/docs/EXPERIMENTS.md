# Experiment log

Day-one measurements from the repo's own bench. Reproduce with the
scripts in `examples/`; all runs are seeded and CPU-only.

## E1 — Knowledge capacity of a small network (2026-07-02)

Miniature of Allen-Zhu & Li ([arXiv:2404.05405](https://arxiv.org/abs/2404.05405)):
N random 8-bit facts (key -> uniform random value) memorized by a fixed
22,848-parameter MLP; knowledge stored is lower-bounded by
8 bits x facts recalled. Script:
`examples/knowledge_capacity_experiment.py` (~5 min CPU).

### Capacity sweep

| facts | recall | bits stored | bits/param |
|------:|-------:|------------:|-----------:|
| 1,000 | 1.000 | 8,000 | 0.35 |
| 2,000 | 1.000 | 16,000 | 0.70 |
| 4,000 | 1.000 | 32,000 | 1.40 |
| 8,000 | 0.563 | 36,064 | **1.58** |
| 16,000 | 0.152 | 19,416 | 0.85 |

Perfect recall up to 4,000 facts; the measured plateau is **~1.6
bits/param** — strikingly close to the ~2 bits/param that Allen-Zhu & Li
report for transformers, on a completely different architecture (a plain
MLP) and five orders of magnitude fewer parameters.

**Honest caveats.** (1) The 16,000-fact run is limited by optimization
(4,000 full-batch steps), not necessarily by capacity — stored bits
*decrease*, which is the classic undertraining signature Allen-Zhu & Li
also document (their capacity law needs ~1000 exposures per fact).
(2) The bits-stored estimator (8 x correct recalls) is a lower bound; it
ignores partial knowledge in the logits. Both caveats push the true
plateau *up*, toward the transformer value.

### Quantization destruction curve (saturated model)

| weight bits | recall | knowledge kept |
|------------:|-------:|---------------:|
| 16 (none) | 0.152 | 100.0% |
| 8 | 0.148 | 97.5% |
| 4 | 0.052 | 34.4% |
| 3 | 0.023 | 15.4% |
| 2 | 0.005 | 3.2% |

Reproduces the shape reported by Allen-Zhu & Li at toy scale: **8-bit
quantization is nearly free (97.5% kept); the cliff is between 8 and 4
bits** for a model filled to capacity. Note this model was *saturated* —
a model below capacity has slack and survives 4-bit much better, which is
exactly why practical 4-bit quantization of undertrained large models
works. The two regimes (full vs. slack) are the interesting frontier.

## E2 — The SGD-channel discriminating experiment (2026-07-02)

Tests the mechanism proposed in CAPACITY_THEORY.md §4: if stored
knowledge is bounded by a Gaussian-channel capacity ½log₂(1+SNR), then
injected gradient noise (std = σ × grad std, per tensor, per step) must
collapse capacity along the log curve. Script:
`examples/sgd_channel_test.py` (N = 6,000 facts, ~8 min CPU).

| σ (rel. noise) | recall | bits stored | bits/param |
|---:|---:|---:|---:|
| 0.0 | 1.000 | 48,000 | **2.101** |
| 1.0 | 0.975 | 46,824 | 2.049 |
| 2.0 | 0.906 | 43,480 | 1.903 |
| 4.0 | 0.808 | 38,768 | 1.697 |

**Finding 1 (the constant crossed 2).** With zero injected noise the
22,848-param MLP stores 48,000 bits at *perfect recall*:
**κ ≥ 2.10 bits/param** — the same region as Allen-Zhu & Li's ~2
bits/param for transformers, now observed on a second architecture from
this repo's own bench. (E1's 1.58 was optimization-limited; the true
plateau is ≥ 2.10 and still a lower bound.)

**Finding 2 (naive channel refuted in magnitude, noise-sensitivity
confirmed in direction).** Capacity does fall monotonically with noise —
but far more slowly than the single-step channel formula predicts
(σ = 4 gives per-step SNR ≈ 1/16, naive κ ≈ 0.09; measured 1.70). The
deficit grows roughly linearly in σ (−0.05, −0.20, −0.40). Interpretation:
training is not a single use of the channel — Adam's time-averaging over
thousands of steps recovers most of the per-step SNR. The correct
mechanism must be a rate–distortion analysis of the whole *trajectory*,
not of one step. The naive model is dead; the refined question is alive
and sharply posed.

## E3 — The pseudo-model / two-walls control (2026-07-02)

Empirical verification of Theorem 8 and formula F2
(docs/INDISTINGUISHABILITY.md). Same fixed-capacity student (~448-bit
logistic regression) distills two teachers on a domain of D = 2^14
inputs; agreement measured under uniform Q. Script:
`examples/pseudomodel_demo.py` (~4 s, pure numpy).

| samples m | random teacher | halfspace teacher |
|----------:|---------------:|------------------:|
| 100 | 0.492 | 0.930 |
| 500 | 0.500 | 0.969 |
| 2,000 | 0.506 | 0.980 |
| 8,000 | 0.501 | 0.988 |
| 20,000 | 0.507 | 0.990 |

**The two walls, separated on one line.** The *incompressible* teacher
(random function, F2 worst case) pins student agreement at chance 0.500
forever — pigeonhole holds exactly, a small responder provably cannot
fake it, and this is why "12 GB cannot store a random 1T function" is a
theorem, not an engineering gap. The *structured* teacher (a halfspace,
small pseudo-dimension) reaches 0.99 indistinguishability with a
tiny student after a few hundred samples — Door 2, and the reason
distillation of real models works. **Same student both columns**; the
only variable is the teacher's structure under Q. This is Theorem 8's
content made visible: storage is not the wall, structure-plus-search is.

### What would make this discovery-grade

1. Precision: is the plateau exactly the same constant across
   architectures (MLP, attention, conv), optimizers and data
   distributions? A universal constant would be a law; spread would be an
   anomaly. Either is publishable.
2. Proper estimator: replace the recall lower bound with the true mutual
   information (logit entropies), and train to the exposure counts the
   capacity law requires.
3. Scale: repeat over 3-4 orders of magnitude of parameters with error
   bars. All of it fits on one GPU-day.
