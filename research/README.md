# research/

Exploratory research notes that sit *adjacent* to AirLLM's mission. AirLLM exists
to make giant Transformer weights fit in small memory; this folder asks the prior
question — whether knowledge must take the form of a giant weight file at all.

## Contents

**Phase 1 — paradigm map (the discipline).**
- [`post-transformer-paradigm.md`](post-transformer-paradigm.md) — a 5-phase,
  deliberately *critical* research program on post-Transformer cognitive
  architectures. It separates what is demonstrated, what is plausible-but-untested,
  and what is rhetoric, and it ends on a single falsifiable prototype rather than a
  manifesto.
- [`experiments/adaptive_compute_probe.py`](experiments/adaptive_compute_probe.py)
  — a measurement *framework* for the Phase 4 hypothesis (adaptive compute budget).
  It ships with a synthetic stand-in backend so the accounting can be reviewed
  without a GPU. **Its numbers are not a result** — replace the synthetic backend
  and eval set (see the `AirLLMBackend` stub) to produce real, citable evidence.

```bash
python3 research/experiments/adaptive_compute_probe.py --n 300
```

**Phase 2 — the measurement engine (the instrument).** Before scaling parameters,
measure intelligence *per unit of compute*.
- [`cognitive-efficiency-engine.md`](cognitive-efficiency-engine.md) — defines the
  Cognitive Efficiency Quotient (CEQ = complexity mastered at quality, per
  kilotoken) and an **ablation protocol** over the configurations *Claude Code
  solo → +memory → +planner → +codegen → +adaptive-compute*. Designed so a
  fashionable addition can *lose*: a graft that adds tokens without enough useful
  output scores a multiplier < 1 and is dropped.
- [`benchmarks/`](benchmarks/) — runnable scorer (`cee.py`), a frozen task suite,
  the three axes (`token_efficiency/`, `task_completion/`, `reasoning_cost/`), and a
  clearly-labeled synthetic run log.
- [`memory_architecture/`](memory_architecture/) — notes on editable memory (`C1`),
  gated: no implementation ships until a *directed* memory config clears a measured
  multiplier > 1.

```bash
python3 research/benchmarks/cee.py --selftest
python3 research/benchmarks/cee.py --runs research/benchmarks/sample_runs.synthetic.jsonl --baseline C0
```

The intended chain: **CEE measures the multiplier → build only capabilities proven
> 1 → only then scale ("AirLLM 2050").** Efficiency per token is the gate; parameter
count is not.

## Status

Speculative / non-load-bearing. Nothing here is imported by the `airllm` package or
changes its behavior. These are notes and a test harness, kept honest: a number
that hasn't been measured is not a result.
