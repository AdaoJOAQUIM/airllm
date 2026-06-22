# Axis: reasoning cost

**Targets the waste of *compute*** — the core of [doc 1, Phase 4](../../post-transformer-paradigm.md).
The defining waste of today's systems: nearly the same effort spent on a trivial
question and on a hard proof.

Stresses the **Q-vs-T curve** on difficulty-heterogeneous tasks. The decisive
reading is not a single score but the *shape*: does a configuration hold quality
while spending less on the easy end (adaptive compute), or does it pay a flat cost
regardless of difficulty?

Tasks tagged `axis: "reasoning_cost"` in [`../tasks.jsonl`](../tasks.jsonl):
`rc-01`..`rc-03` (3-step arithmetic → multi-constraint scheduling → short
combinatorial proof), spanning complexity 0.40 → 0.95.

This axis is the bridge to the [`adaptive_compute_probe`](../../experiments/adaptive_compute_probe.py):
that probe isolates the mechanism (budget allocation per request); this axis checks
whether the mechanism actually moves CEQ once embedded in a full system.

Key question: does `C4` (+adaptive compute) keep `Q` while lowering `T` mostly on
the low-complexity items? If the token savings come from the *hard* items, quality
is being quietly sacrificed — inspect per-task, not just the mean.
