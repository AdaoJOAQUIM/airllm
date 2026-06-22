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

**Phase 3 — the closed loop + orchestration (the instrument acts).** The CEE only
observes/ranks; this layer *acts on real data, observes the real result, and turns a
minimal intention into a running, validated system*. All multipliers measured from
**real executions** (not a synthetic log), and **variable by architecture — never a
fixed equivalence**.
- [`tme/tme.py`](tme/tme.py) — Token Multiplication Engine: closed loop over a
  bounded program-synthesis domain (real action, field feedback, directed memory
  compression, value routing). Measured: directed memory beats cold by **x1.24**
  raw compute (~x4 CEQ) at equal correctness; density routing beats the value-only
  trap by **x1.90**. Two layers tested and **dropped for failing their own metric**:
  self-reconfiguration (x0.84, hurts) and value-only routing (the trap).
- [`tme/orchestrator.py`](tme/orchestrator.py) — hierarchical orchestrator: a
  compact intention → decompose → synthesize → execute → **validate on held-out**
  (Q=1.00) → learn. Measured: decomposition x1.5, execution amplification ~x6 real
  leaf actions per intention atom, and a learning loop where a repeated batch
  collapses **x5.8** once learned.

```bash
python3 research/tme/tme.py --demo
python3 research/tme/orchestrator.py --demo
```

The intended chain: **CEE measures the multiplier → TME closes the loop and proves
reuse cuts real compute → the orchestrator turns intention into system → build only
capabilities proven > 1 → only then scale ("AirLLM 2050").** Efficiency per token is
the gate; parameter count is not. The honest next step is to swap the DSL solver for
an LLM proposer, keeping loop, directed memory, router, validation, and CEE scoring
identical, so the same measured machinery applies to real token cost.

## Status

Speculative / non-load-bearing. Nothing here is imported by the `airllm` package or
changes its behavior. These are notes and a test harness, kept honest: a number
that hasn't been measured is not a result.
