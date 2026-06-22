# Axis: token efficiency

**Targets the waste of *tokens*.** How much useful cognitive output per kilotoken?

Stresses the upper links of the value chain — `useful_decisions / (T/1000)`
(frugality) and `Q` — on tasks where the *answer* is cheap but systems often
over-spend (verbose reasoning on trivial items, redundant re-reading).

Tasks tagged `axis: "token_efficiency"` in [`../tasks.jsonl`](../tasks.jsonl):
`te-01`..`te-03` (capital lookup, CSV→JSON reformat, one-sentence summary).

Key question for this axis: does a capability *reduce* `T` on easy tasks, or does it
pay a fixed overhead it can never recover when the task is trivial? This is where
"smart" additions most often score < 1.

Score: `python3 ../cee.py --runs <your_runs.jsonl> --baseline C0` then read the
`frugal=` column for tasks on this axis.
