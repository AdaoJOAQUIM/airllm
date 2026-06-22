# research/

Exploratory research notes that sit *adjacent* to AirLLM's mission. AirLLM exists
to make giant Transformer weights fit in small memory; this folder asks the prior
question — whether knowledge must take the form of a giant weight file at all.

## Contents

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

## Status

Speculative / non-load-bearing. Nothing here is imported by the `airllm` package or
changes its behavior. These are notes and a test harness, kept honest: a number
that hasn't been measured is not a result.
