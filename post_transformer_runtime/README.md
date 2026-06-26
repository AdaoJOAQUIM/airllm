# Post-Transformer Neural Runtime

Research scaffold exploring the project's central, falsifiable question:

> Are trillions of parameters a fundamental necessity, or merely an inefficient
> representation of intelligence?

Starting point: the **AirLLM** engine in this repository. The Phase 0 audit
(`../docs/AIRLLM_AUTOPSY.md`) establishes — with code references — that AirLLM
reduces *peak memory* by streaming weights layer-by-layer from disk, but reduces
**neither** parameter count, FLOPs, nor (absent quantization) storage. It is the
ceiling of the dense-weights paradigm. This project tries to go under that
ceiling.

## Status

| Phase | Item | State |
|---|---|---|
| 0 | Scientific autopsy of AirLLM (`docs/AIRLLM_AUTOPSY.md`) | ✅ done |
| 5 (foundation) | Reproducible benchmark harness (`benchmarks/`) | ✅ done, self-tested |
| 6 (Axis 6) | Sparsity / MoE prototype (anchored on `airllm_mixtral.py`) | ⬜ not started |
| 1–4, 7–9 | Hypernetworks, neural fields, MDL, VQ, associative memory, … | ⬜ not started |

Why the benchmark harness *before* the research axes: nothing downstream can be
honestly claimed without a baseline to beat. The harness fixes that baseline and
makes every future axis measurable on the same six axes (memory / VRAM / storage
/ throughput / energy / quality). See `benchmarks/README.md`.

## Method (non-negotiable)

Every axis gets a `REPORT.md` with: hypothesis → protocol → prototype →
benchmark (via this harness) → result → limits → verdict
(**VALIDATED / PARTIAL / REFUTED / NOT TESTED**). No verdict without a
reproducible benchmark. Untested ≠ validated.

The lowest-risk first experiment, per the autopsy, is **Axis 6 (sparsity)**:
`airllm_mixtral.py` already routes to experts; streaming only the *active*
experts per token (instead of the whole layer) is a concrete, measurable test of
"density is redundancy" — with an anchor already in the codebase.

## Layout

```
post_transformer_runtime/
├── benchmarks/        # the measuring instrument (done)
│   ├── metrics.py     #   stdlib-only measurement primitives
│   ├── runner.py      #   engine-agnostic run_benchmark()
│   ├── baseline_airllm.py  # AirLLM adapter + CLI (the baseline)
│   ├── report.py
│   ├── selftest.py    #   runs without torch/GPU/downloads
│   └── results/       #   real-hardware measurements only
└── (future axis modules: sparse_engine/, quantization/, hypernetwork/, ...)
```
