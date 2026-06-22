# Axis: task completion

**Targets the waste of *context / coordination*.** Does the system turn reasoning
into successful external effects across multi-step, multi-file work?

Stresses the lower links of the value chain — `actions_succeeded / actions_attempted`
(action conversion) and `Q` — on tasks where the bottleneck is not "thinking" but
keeping many moving parts coherent (edits that pass tests, no regressions).

Tasks tagged `axis: "task_completion"` in [`../tasks.jsonl`](../tasks.jsonl):
`tc-01`..`tc-03` (add a CLI flag + tests, fix a 3-file integration, refactor away a
circular import with the API and suite kept green).

Key question for this axis: does a capability (memory, planner, codegen) raise
action conversion enough to justify the tokens it spends coordinating? A planner
that cuts wasted decisions should shine here; raw memory often does not.

Score: read the `act=` and `Q=` columns for tasks on this axis.
