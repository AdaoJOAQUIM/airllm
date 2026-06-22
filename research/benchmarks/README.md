# CEE benchmarks

The scoring engine and task suite for the [Cognitive Efficiency Engine](../cognitive-efficiency-engine.md).
The engine **measures** systems; it does not run them. It consumes run logs and
emits the CEQ report plus the ablation that decides whether a capability multiplies
efficiency or just adds token overhead.

## Quick start

```bash
python3 cee.py --selftest                                   # verify the arithmetic
python3 cee.py --runs sample_runs.synthetic.jsonl --baseline C0   # demo report (synthetic!)
```

Any log whose filename contains `synthetic` prints a warning banner — its numbers
test the math, not any real system.

## Files

- `cee.py` — scoring core: CEQ, value-chain metrics, marginal ablation, self-test.
- `tasks.jsonl` — **frozen** task suite. `complexity` (K) and `human_baseline_s`
  (H_b) live here and are never self-reported by the system under test.
- `sample_runs.synthetic.jsonl` — invented run log for demonstration only.
- `token_efficiency/`, `task_completion/`, `reasoning_cost/` — the three axes
  (one README each) describing what each stresses and which tasks belong to it.

## Run-log schema (one JSON object per line)

| field | meaning |
|---|---|
| `task_id`, `config` | which task, which configuration (`C0`..`C4`) |
| `tokens_in`, `tokens_out` | **all** tokens, incl. memory reads / planner / validation |
| `useful_decisions`, `total_decisions` | judged via the rubric (blind to config) |
| `actions_succeeded`, `actions_attempted` | external effects (tests pass, cmd advances) |
| `solution_quality` | Q in [0,1] against the task's frozen `success_criterion` |
| `complexity`, `human_baseline_s` | copied from the frozen task spec |
| `oversight_s` | human seconds spent supervising/correcting this run |

## Producing REAL runs (the next step)

Nothing here is a result yet. To get real multipliers you must instrument actual
sessions:

1. Run each task under each config (`C0` baseline … `C4` +adaptive-compute).
2. Capture total token usage from the provider/runtime (not an estimate).
3. Score `Q` against the frozen `success_criterion`; have a second person judge
   `useful_decisions` blind to which config produced the transcript.
4. Append one record per run to a `runs.jsonl` and score it with `cee.py`.

The honesty rule that makes the benchmark worth anything: **`T` counts every token
the configuration spent**, including the overhead of the very capability you hope
will win. A multiplier that ignores its own cost is a lie.
