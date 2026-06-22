# memory_architecture/

Notes on the **editable memory** module — candidate `C1` in the CEE ablation and
component "B" of the target architecture in
[`post-transformer-paradigm.md`](../post-transformer-paradigm.md) (Phase 5).

## Why this is a folder of notes, not code yet

The CEE exists precisely to stop us from building memory because memory is
fashionable. On the synthetic demo log, *undirected* memory (`C1`) scores a
multiplier **< 1** — it adds tokens (more context to re-read) without enough extra
useful decisions to pay for itself. That is the expected failure mode and the whole
point of measuring first.

The hypothesis from doc 1 is sharper:

> Raw memory has a multiplier ≤ 1 **until it is directed by a planner** (`C2`).
> Memory's value is not storage; it is *retrieval the planner actually uses*.

So the design question is not "what database?" but:

1. **What gets written?** Not raw transcript — distilled, addressable units
   (decisions made, facts established, dead ends ruled out).
2. **What gets read, and who decides?** The planner issues a retrieval plan; memory
   is pulled on demand, bounded, not relayed wholesale into context every turn.
3. **What gets forgotten?** Bounded size with eviction by predicted future utility
   — the hardest part, and exactly where Neural Turing Machines failed (doc 1,
   candidate A). Editable/forgetting memory is deferred until directed memory itself
   clears multiplier > 1.

## Build order (gated by measurement)

```
  measure C1 (raw memory) ........... expected <= 1  (do NOT ship on its own)
        │
        ▼
  measure C1+C2 (memory directed by planner)
        │  ship only if multiplier > 1
        ▼
  add bounded eviction / forgetting (candidate A) ... only then
```

No memory implementation lands in `airllm` from this folder until a directed
configuration clears a measured multiplier > 1 on the CEE suite. Notes only, by
design.
