# Token Multiplication Engine (TME) — closed loop

The answer to the critique that the CEE only *observes/evaluates/ranks*. This engine
**acts, observes the real result, and rewrites itself**. The multiplier is measured
from **real executions**, not a synthetic log.

```bash
python3 tme.py --selftest     # verify loop + reuse + multiplier
python3 tme.py --demo         # measured cold-vs-warm multiplier + value routing
python3 tme.py --emit runs.jsonl && python3 ../benchmarks/cee.py --runs runs.jsonl --baseline cold
```

## What is real here (and what is bounded)

Real: the closed loop, the execution of candidate programs on real inputs, and every
number reported (cost = candidates actually evaluated; correctness checked for real).
Bounded: the "solver" is search over a small DSL, **not an LLM**. The claim is narrow
and true: *pattern reuse + self-reconfiguration cut real compute at equal
correctness*, and that reduction is the measured multiplier.

## The five layers the critique demanded — all wired, all measured

| Layer | Where | Measured signal (from `--demo`) |
|---|---|---|
| Real action (closed loop) | `run_program` executes candidates | solved 11/11 on real test cases |
| Field feedback | `Engine.solve` counts real cost | cost in candidates evaluated |
| Self-reconfiguration | `_win` reorders `op_order` by wins | op priority rewritten between cycles |
| Value routing | `value_routing` (value ÷ est-cost) | density routing beats value-only **x1.90** |
| Memory compression | bounded `memory` (sig → patterns) | 11 solves → 5 patterns (2.2 solves/pattern) |

## The measured result

```
cold (no memory/reconfig): solved 11/11  cost=270 candidates
warm (memory + reconfig) : solved 11/11  cost=209 candidates
MULTIPLIER (equal correctness) = x1.29        # raw compute
CEE CEQ multiplier (cold→warm) = x4.09        # complexity-weighted, via cee.py
```

Two honest caveats that matter:

1. **Value-only routing is a trap** (it captured 10 vs 19 under a tight budget). The
   engine routes by **value/est-cost**; sorting by value alone is exactly the
   "optimize the wrong tasks" failure. The demo shows both, on purpose.
2. The multiplier grows with **reuse density** (instances per pattern family). On a
   suite with no repeated structure it tends to 1 — memory cannot multiply what it
   has never seen. That is the correct, non-magical behavior.

## How this closes the loop the CEE left open

```
  CEE:  observe -> evaluate -> rank                 (measures intelligence)
  TME:  act -> observe real result -> compress -> reconfigure -> cheaper next cycle
                                                    (restructures itself, in a loop)
```

Next real step toward "post-AirLLM": swap the DSL solver for an LLM proposer (keep
the loop, execution, memory, router, and CEE scoring identical), so the same
measured multiplier applies to real token cost instead of candidate count.
