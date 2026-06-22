# Token Multiplication Engine (TME) — closed loop + orchestration

The answer to the critique that the CEE only *observes/evaluates/ranks*. This layer
**acts, observes the real result, and rewrites itself**, and an orchestrator turns a
**minimal intention into a running, validated system**. Every multiplier is measured
from **real executions**, never a fixed equivalence.

```bash
python3 tme.py --selftest            # closed loop: reuse + multiplier
python3 tme.py --demo                # cold-vs-warm multiplier + value routing
python3 orchestrator.py --selftest   # intention -> system, validated on held-out
python3 orchestrator.py --demo       # decomposition, amplification, learning loop
python3 tme.py --emit runs.jsonl && python3 ../benchmarks/cee.py --runs runs.jsonl --baseline cold
```

## The corrected paradigm (important)

There is **no fixed equivalence** like "1 token = N actions". Amplification is
*variable, produced by architecture* (abstraction + decomposition + automation +
memory), and it **collapses toward 1 when there is nothing to reuse** — memory
cannot multiply the unseen. The code is built to show exactly this, including where
a fashionable layer *fails its own metric*.

## What is real (and what is bounded)

Real: the closed loop, candidate programs actually executed on real inputs, held-out
validation, and every reported number (cost = candidates evaluated; correctness
checked for real). Bounded: the "solver" is search over a small DSL, **not an LLM**.
The claim is narrow and true: *directed memory + hierarchical orchestration cut real
compute at equal, validated correctness*.

## Two honest negative results we kept (not hid)

1. **Self-reconfiguration does NOT help here.** Reordering op priority by wins —
   even conservatively — pushes a still-needed op down the search order and makes
   later tasks cost more (memory+reconfig = x0.84). So it is **measured and left
   OFF**; the multiplier rests on memory alone. Same discipline as the CEE: a layer
   that fails its metric gets dropped, however appealing.
2. **Value-only routing is a trap.** Sorting by value alone burns the budget on
   expensive tasks (captured 10 vs 19). The router uses **value ÷ estimated cost**;
   density routing beats value-only **x1.90**. The demo shows both on purpose.

## Measured results

Closed loop (`tme.py --demo`):
```
cold (no memory): 11/11 solved, cost=270 candidates
warm (memory)   : 11/11 solved, cost=217 candidates
MULTIPLIER (equal correctness) = x1.24    # raw compute (CEQ-weighted x~4 via cee.py)
reconfiguration probe = x0.84             # tested, does not help, left OFF
```

Orchestration (`orchestrator.py --demo`) — minimal intention → running system:
```
intention "double | evens | runsum > summary"  (4 atoms)
  -> 6 synthesized sub-programs (decomposition x1.5)
  -> x6 real leaf actions / atom (execution amplification)
  -> held-out validation Q=1.00            # the generated system generalizes
L5 learning loop: same intentions, 2nd pass collapses x5.8 (the system learned)
```

## The five levels of cognitive leverage — all wired, all measured

| Level | Where | Measured signal |
|---|---|---|
| L1 expand intention | `parse` | compact spec → structured objective |
| L2 decompose | `GROUPS` fan-out | one verb → up to 3 sub-tasks (decomposition x1.5) |
| L3 generate | `Engine.solve` (DSL search) | real programs synthesized per sub-task |
| L4 automate + test | `_validate` on held-out | Q=1.00 end-to-end generalization |
| L5 improvement loop | shared `Engine.memory` | 2nd pass collapses x5.8 |
| (router) value routing | `value_routing` | value/cost beats value-only x1.90 |
| (compression) memory | bounded `memory` (directed signature) | solves compressed to bounded patterns |

## How it closes the loop the CEE left open

```
  CEE:           observe -> evaluate -> rank                 (measures intelligence)
  TME:           act -> observe real result -> compress -> cheaper next cycle
  Orchestrator:  minimal intention -> decompose -> generate -> execute -> validate
                 -> learn  (a small intention programs a whole running system)
```

## Intent → Execution Graph Compiler (`igc.py`)

The piece between an *optimizer* and a *cognitive orchestrator*. Instead of a linear
pipeline, it compiles an intention into a **causal action graph** (nodes = actions,
edges = dependencies, weights = value/cost) and **allocates compute by impact**.

The real multiplier is **depth of compilation**, not actions-per-token: a graph lets
a shared sub-result execute **once** and feed many consumers (a list cannot), and
lets a budget **prune low-impact nodes** by value-density.

```bash
python3 igc.py --demo        # graph reuse + value-per-budget vs linear baseline
```

Measured (`igc.py --demo`):
```
diamond intention (6 atoms, goals share `x` and `e`):
  shared-node reuse x1.33 under budget (x2.0 unconstrained — selftest)
  value captured per budget: graph x1.25 vs linear recompute-per-goal
  held-out Q=1.00
linear intention (nothing to share):
  reuse x1.00 — amplification collapses toward 1, structure-dependent (correct)
```

This is the concrete answer to "it doesn't turn an intention into a *system*": a
compact intention is compiled to a dependency graph, scheduled under a compute
budget by value/cost, synthesized, executed, and validated — and the graph's
advantage over a list is **measured**, not asserted.

## The honest next step toward "post-AirLLM"

Swap the DSL solver for an **LLM proposer**, keeping the loop, directed memory,
router, validation, and CEE scoring identical. Then the *same measured machinery*
applies to real token cost: a compact intention drives an LLM-synthesized,
executed, validated, and self-cheapening system — variable amplification, measured,
not promised.
