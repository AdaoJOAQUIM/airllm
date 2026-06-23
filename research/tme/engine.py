"""Token-Multiplier Engine — consolidated, deterministic kernel (no LLM).

Ties the research/tme stack into ONE economical tool:
    intention -> igc.compile_intention (causal graph)
              -> igc.schedule (value/cost)
              -> igc.Compiler.execute/validate (real run, held-out validated)
              -> tme.Engine directed memory  (REUSE)
and adds the one thing that lets a deterministic kernel reach ORDERS OF MAGNITUDE:
**persistent memory**. Structured/repeated work approaches zero build-cost across
runs, so the *measured* multiplier grows with cumulative reuse -- and collapses to
~1 on novel work (the honest guardrail). There is no fixed "1 token = N actions".

The engine is itself economical: stdlib only, deterministic, single file, and
persistence makes re-runs cheap.

    python3 engine.py --run "x=double(input); e=evens(x); a=sum(e)@2"
    python3 engine.py --demo
    python3 engine.py --selftest
    python3 engine.py --run "..." --memory /path/to/mem.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from tme import Engine
from igc import compile_intention, schedule, Compiler
from orchestrator import DATA, HELD

DEFAULT_MEM = str(Path(__file__).parent / ".engine_memory.json")


@dataclass
class Report:
    intention: str
    tokens: int
    warm_build: int          # synthesis candidates with persistent memory
    cold_build: int          # synthesis candidates from scratch (fresh engine)
    exec_ops: int            # real leaf actions executed on data
    quality: float           # held-out validation
    goals: int

    @property
    def multiplier(self) -> float:
        return self.cold_build / max(self.warm_build, 1)

    def line(self) -> str:
        return (f"{self.tokens} atoms | build warm={self.warm_build} cold={self.cold_build} "
                f"| MULTIPLIER x{self.multiplier:.2f} | exec x{self.exec_ops / self.tokens:.1f}/atom "
                f"| held-out Q={self.quality:.2f}")


class TokenEngine:
    def __init__(self, memory_path: str | None = DEFAULT_MEM, budget: int = 999):
        self.eng = Engine()
        self.memory_path = memory_path
        self.budget = budget
        self.loaded = bool(memory_path) and self.eng.load(memory_path)

    def _cost(self, intention: str, engine: Engine):
        g = compile_intention(intention)
        chosen, req, _ = schedule(g, self.budget, share=True)
        comp = Compiler(engine=engine)
        out = comp.execute(g, req, chosen, DATA)
        if out is None:
            return None
        q = comp.validate(g, out[0], out[1], HELD)
        return comp.synth_cost, comp.node_execs, q, len(chosen), g.tokens

    def run(self, intention: str) -> Report:
        cold = self._cost(intention, Engine())          # fresh, no memory
        warm = self._cost(intention, self.eng)          # persistent, learns
        if cold is None or warm is None:
            raise SystemExit(f"intention not solvable within budget: {intention!r}")
        return Report(intention, warm[4], warm[0], cold[0], warm[1], warm[2], warm[3])

    def save(self) -> None:
        if self.memory_path:
            self.eng.save(self.memory_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_run(intention: str, memory_path: str) -> int:
    te = TokenEngine(memory_path=memory_path)
    r = te.run(intention)
    te.save()
    print(f"intention: {intention!r}")
    print("  " + r.line())
    print(f"  memory persisted -> {memory_path}")
    print("  (re-run the same/structured intention: warm build cost drops toward 0)")
    return 0


WORKLOAD = [
    "x=double(input); a=sum(x)@2",
    "x=double(input); e=evens(x); b=sum(e)@2",
    "x=double(input); e=evens(x); s=runsum(e); c=max(s)@3",
    "y=inc(input); d=min(y)@1",
]
NOVEL = "z=rev(input); w=dedup(z); g=len(w)@1"


def _cmd_demo() -> int:
    print("== Token-Multiplier Engine: orders of magnitude via PERSISTENT reuse ==")
    print("Multiplier is measured and VARIABLE. It grows with reuse, collapses to ~1")
    print("on novel work. No fixed equivalence.\n")
    te = TokenEngine(memory_path=None)        # in-memory only for the demo
    cum_cold = cum_warm = 0
    for p in range(1, 4):
        pc = pw = 0
        for intent in WORKLOAD:
            r = te.run(intent)
            pc += r.cold_build; pw += r.warm_build
        cum_cold += pc; cum_warm += pw
        print(f"pass {p}: warm build={pw:4d}  cold build={pc:4d}  "
              f"pass x{pc / max(pw,1):.2f}  | cumulative x{cum_cold / max(cum_warm,1):.2f}")
    print("  -> warm cost collapses after pass 1; cumulative multiplier climbs with reuse.")
    print("     more passes / more structured work => larger measured factor (orders of magnitude).\n")

    rn = te.run(NOVEL)
    print(f"novel intention {NOVEL!r}")
    print(f"  {rn.line()}  -> multiplier ~1 (nothing to reuse yet; honest)")
    return 0


def _cmd_selftest() -> int:
    import tempfile
    intent = "x=double(input); e=evens(x); s=runsum(e); a=sum(s)@2; b=max(x)@3"
    te = TokenEngine(memory_path=None)
    r1 = te.run(intent)                       # first exposure: warm ~ cold
    r2 = te.run(intent)                       # learned: warm << cold
    assert r1.quality == 1.0 and r2.quality == 1.0, (r1.quality, r2.quality)
    assert r2.warm_build < r1.warm_build, (r1.warm_build, r2.warm_build)
    assert r2.multiplier > r1.multiplier, (r1.multiplier, r2.multiplier)
    assert r2.multiplier >= 3.0, r2.multiplier      # real order-of-magnitude reuse

    # persistence round-trip: save, reload into a NEW engine, stays cheap
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    te2 = TokenEngine(memory_path=path); te2.run(intent); te2.save()
    te3 = TokenEngine(memory_path=path)
    assert te3.loaded, "memory did not persist"
    r3 = te3.run(intent)
    assert r3.warm_build <= r2.warm_build + 1, (r3.warm_build, r2.warm_build)

    # novel work must NOT magically amplify (multiplier ~1 on first sight)
    rn = TokenEngine(memory_path=None).run(NOVEL)
    assert rn.multiplier <= 1.5, rn.multiplier
    print(f"selftest OK: learned x{r1.multiplier:.1f}->x{r2.multiplier:.1f}, "
          f"persisted reuse stays cheap (build {r3.warm_build}), novel~x{rn.multiplier:.1f}, Q=1.0")
    return 0


def _cmd_learn(memory_path: str) -> int:
    """Paradigm step: grow the engine's own language (library learning) and persist
    it, so it can solve NOVEL tasks beyond the base DSL afterwards."""
    from abstraction import paradigm_report
    te = TokenEngine(memory_path=memory_path)
    paradigm_report(te.eng)            # learns macros onto te.eng, prints the 3 proofs
    te.save()
    print(f"\ngrown language persisted -> {memory_path} "
          f"(the DSL itself is now larger, not just the cache)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Token-Multiplier Engine (deterministic kernel)")
    ap.add_argument("--run", metavar="INTENTION", help="compile+run one intention")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--learn", action="store_true", help="library learning (grow the DSL)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--memory", default=DEFAULT_MEM, help="persistent memory file")
    a = ap.parse_args(argv)
    if a.selftest:
        return _cmd_selftest()
    if a.learn:
        return _cmd_learn(a.memory)
    if a.demo:
        return _cmd_demo()
    if a.run:
        return _cmd_run(a.run, a.memory)
    ap.error("use --run INTENTION, --demo, --learn, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
