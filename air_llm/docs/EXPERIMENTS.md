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
