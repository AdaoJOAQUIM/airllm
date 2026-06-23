"""Token Multiplication Engine (TME) — closed-loop, real execution.

Answers the critique that the CEE only *measures*: this engine ACTS, OBSERVES the
real result, and REWRITES itself. The multiplier here is not theoretical — it is
measured from real program executions in a bounded synthesis domain.

Closed loop per task:
    propose candidate  ->  EXECUTE it for real on test cases  ->  observe pass/fail
    ->  on fail: search/correct   ->  on pass: COMPRESS into a reusable pattern
    ->  RECONFIGURE the engine's own search policy from observed successes

Five layers the critique asked for, all real & measured (not simulated):
  1. Real action      : candidate programs are actually run on real inputs.
  2. Field feedback   : cost = candidates actually evaluated; correctness = real.
  3. Self-reconfig    : op order is rewritten by success frequency between cycles.
  4. Value routing    : budget allocated by task value & solvability under a cap.
  5. Memory compression: solutions distilled to (signature -> program) patterns,
                         bounded; reuse cuts real cost on later tasks.

Honest scope: the "solver" is search over a small DSL, not an LLM. What is real is
the closed loop, the execution, and the measurement. The claim is narrow and true:
*pattern reuse + self-reconfiguration reduce real compute at equal correctness*,
and the reduction is the measured multiplier.

    python3 tme.py --selftest     # verify the loop + math
    python3 tme.py --demo         # measured cold-vs-warm multiplier + value routing
    python3 tme.py --emit runs.jsonl   # write CEE-scorable records, then:
    python3 ../benchmarks/cee.py --runs runs.jsonl --baseline cold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

Prog = tuple[tuple[str, ...], str | None]  # (base op names, optional terminal)


# --------------------------------------------------------------------------- #
# DSL: list->list base ops, list->scalar terminals. Real, pure functions.
# --------------------------------------------------------------------------- #
def _take(k): return lambda xs: xs[:k]
def _drop(k): return lambda xs: xs[k:]
def _add(k): return lambda xs: [x + k for x in xs]
def _mul(k): return lambda xs: [x * k for x in xs]
def _prefix(xs):
    out, s = [], 0
    for x in xs:
        s += x; out.append(s)
    return out
def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

BASE: dict[str, Callable[[list[int]], list[int]]] = {
    "rev": lambda xs: xs[::-1],
    "srt": lambda xs: sorted(xs),
    "srtd": lambda xs: sorted(xs, reverse=True),
    "psum": _prefix,
    "dedup": _dedup,
    "feven": lambda xs: [x for x in xs if x % 2 == 0],
    "fodd": lambda xs: [x for x in xs if x % 2 == 1],
    **{f"add{k}": _add(k) for k in (-3, -2, -1, 1, 2, 3)},
    **{f"mul{k}": _mul(k) for k in (2, 3, -1)},
    **{f"take{k}": _take(k) for k in (1, 2, 3)},
    **{f"drop{k}": _drop(k) for k in (1, 2, 3)},
}
TERMINALS: dict[str, Callable[[list[int]], int]] = {
    "sum": sum, "max": max, "min": min, "len": len,
}
MAXD = 2  # search depth; the demo tasks are solvable within this.


def run_program(prog: Prog, x: list[int]):
    """REAL execution of a candidate on one input. Returns output or _ERR."""
    names, term = prog
    val: list[int] = list(x)
    try:
        for n in names:
            val = BASE[n](val)
        if term is not None:
            return TERMINALS[term](val)
        return val
    except Exception:  # empty max/min etc. -> non-match, counts as a real failure
        return _ERR

_ERR = object()


# --------------------------------------------------------------------------- #
# Tasks: families repeat (different numbers, same structure) so memory can pay off.
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    tid: str
    family: str
    value: float
    pairs: list[tuple[list[int], object]]  # real (input, expected output) cases
    est: float = 1.0                        # DECLARED cost prior (not an oracle)

    @property
    def scalar(self) -> bool:
        return isinstance(self.pairs[0][1], int)

    @property
    def density(self) -> float:             # value per expected unit of compute
        return self.value / self.est


# Declared cost priors per family (cheap ops vs search-heavy ops). Estimates, not
# the real measured cost -- that is what a router must work with in practice.
_EST = {"scale": 8, "reverse": 4, "prefix": 6, "evensum": 40, "topk": 120}

def _mk(tid, fam, val, fn, inputs):
    return Task(tid, fam, val, [(xs, fn(xs)) for xs in inputs], est=_EST[fam])

def task_suite() -> list[Task]:
    I = [[1, 2, 3], [4, 5], [2, 7, 1, 8], [3, 3, 9]]
    J = [[5, 1, 4], [9, 2], [6, 6, 1], [7, 0, 2, 2]]
    return [
        _mk("mul2-a", "scale", 2.0, lambda xs: [x * 2 for x in xs], I),
        _mk("mul2-b", "scale", 2.0, lambda xs: [x * 2 for x in xs], J),
        _mk("mul2-c", "scale", 2.0, lambda xs: [x * 2 for x in xs], [[10], [1, 1, 1]]),
        _mk("rev-a", "reverse", 1.0, lambda xs: xs[::-1], I),
        _mk("rev-b", "reverse", 1.0, lambda xs: xs[::-1], J),
        _mk("psum-a", "prefix", 3.0, _prefix, I),
        _mk("psum-b", "prefix", 3.0, _prefix, J),
        _mk("evsum-a", "evensum", 5.0, lambda xs: sum(x for x in xs if x % 2 == 0), I),
        _mk("evsum-b", "evensum", 5.0, lambda xs: sum(x for x in xs if x % 2 == 0), J),
        _mk("top2-a", "topk", 4.0, lambda xs: sorted(xs, reverse=True)[:2], I),
        _mk("top2-b", "topk", 4.0, lambda xs: sorted(xs, reverse=True)[:2], J),
    ]


def signature(t: Task) -> tuple:
    """Compressed task fingerprint for pattern lookup (not the raw data).

    Behavioral, solution-independent features so different transforms land in
    DIFFERENT buckets -> memory is *directed*, not a colliding bag. Undirected
    (too-coarse) signatures make reuse cost more than it saves (the CEE lesson).
    """
    if t.scalar:
        return ("scalar",)
    xs, ys = t.pairs[0]
    bucket = "same" if len(ys) == len(xs) else ("short" if len(ys) < len(xs) else "long")
    perm = sorted(ys) == sorted(xs)                       # reordering vs value-change
    desc = len(ys) > 1 and ys == sorted(ys, reverse=True)
    first_eq = bool(xs) and bool(ys) and ys[0] == xs[0]
    last_eq = bool(xs) and bool(ys) and ys[-1] == xs[-1]
    return ("list", bucket, perm, desc, first_eq, last_eq)


# --------------------------------------------------------------------------- #
# Candidate generation (search frontier), ordered by a mutable op policy.
# --------------------------------------------------------------------------- #
def candidates(scalar: bool, op_order: list[str]):
    if scalar:
        for term in TERMINALS:                       # depth-0 reduce
            yield ((), term)
        for d in range(1, MAXD):
            for combo in product(op_order, repeat=d):
                for term in TERMINALS:
                    yield (combo, term)
    else:
        for d in range(1, MAXD + 1):
            for combo in product(op_order, repeat=d):
                yield (combo, None)


# --------------------------------------------------------------------------- #
# Engine: memory (compression) + reconfiguration, closed loop.
# --------------------------------------------------------------------------- #
@dataclass
class Engine:
    op_order: list[str] = field(default_factory=lambda: list(BASE))
    memory: dict[tuple, list[Prog]] = field(default_factory=dict)  # bounded patterns
    op_wins: dict[str, int] = field(default_factory=dict)
    mem_cap: int = 6                                  # max patterns per signature
    solved_count: int = 0

    def _passes(self, prog: Prog, t: Task) -> bool:
        return all(run_program(prog, xs) == ys for xs, ys in t.pairs)

    def solve(self, t: Task, budget: int, use_memory: bool, reconfigure: bool):
        """Run the closed loop on one task. Returns (solved, cost, prog)."""
        cost = 0
        sig = signature(t)
        # 1) try compressed memory first (cheap reuse)
        if use_memory:
            for prog in self.memory.get(sig, []):
                if cost >= budget:
                    return False, cost, None
                cost += 1
                if self._passes(prog, t):
                    return self._win(t, sig, prog, cost, reconfigure, from_mem=True)
        # 2) real search
        for prog in candidates(t.scalar, self.op_order):
            if cost >= budget:
                break
            cost += 1
            if self._passes(prog, t):
                if use_memory:
                    self._store(sig, prog)
                return self._win(t, sig, prog, cost, reconfigure, from_mem=False)
        return False, cost, None

    def _store(self, sig, prog):
        bucket = self.memory.setdefault(sig, [])
        if prog in bucket:
            bucket.remove(prog)
        bucket.insert(0, prog)                        # most-recent first
        del bucket[self.mem_cap:]                      # bounded compression

    def _win(self, t, sig, prog, cost, reconfigure, from_mem):
        self.solved_count += 1
        for n in prog[0]:
            self.op_wins[n] = self.op_wins.get(n, 0) + 1
        if reconfigure:                                # the engine rewrites itself
            # Conservative: promote *used* ops as a block, preserving original order
            # within groups (stable). Aggressive full-sort can bury a still-needed
            # op and make later tasks COST MORE -- a real self-modification hazard.
            self.op_order.sort(key=lambda n: self.op_wins.get(n, 0) > 0, reverse=True)
        return True, cost, prog

    def compression_ratio(self) -> float:
        patterns = sum(len(v) for v in self.memory.values())
        return self.solved_count / patterns if patterns else 0.0

    # --- persistence (opt-in): lets reuse accumulate ACROSS runs/sessions -------
    def save(self, path) -> None:
        """Serialize directed memory + stats to JSON. Tuples -> lists."""
        import json as _json
        from pathlib import Path as _Path
        data = {
            "memory": [[list(sig), [[list(p[0]), p[1]] for p in progs]]
                       for sig, progs in self.memory.items()],
            "op_wins": self.op_wins,
            "op_order": self.op_order,
            "solved_count": self.solved_count,
        }
        _Path(path).write_text(_json.dumps(data))

    def load(self, path) -> bool:
        """Load persisted memory if present. Returns True if loaded."""
        import json as _json
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists() or not p.read_text().strip():
            return False
        try:
            data = _json.loads(p.read_text())
        except _json.JSONDecodeError:
            return False
        self.memory = {tuple(sig): [(tuple(names), term) for names, term in progs]
                       for sig, progs in data.get("memory", [])}
        self.op_wins = data.get("op_wins", {})
        self.op_order = data.get("op_order") or list(BASE)
        self.solved_count = data.get("solved_count", 0)
        return True


# --------------------------------------------------------------------------- #
# Value routing: allocate a global budget by value & solvability.
# --------------------------------------------------------------------------- #
def run_suite(tasks, *, use_memory, reconfigure, per_task_budget):
    eng = Engine()
    total_cost, solved, records = 0, 0, []
    for t in tasks:
        ok, cost, prog = eng.solve(t, per_task_budget, use_memory, reconfigure)
        total_cost += cost
        solved += int(ok)
        records.append((t, ok, cost))
    return eng, total_cost, solved, records


def value_routing(tasks, global_budget, per_task_cap):
    """Allocate a hard global budget. Compares three policies; returns value+cost.

    The point: ordering by value ALONE is the classic trap (spends the budget on
    expensive high-value tasks). Ordering by value/est-cost (density) is correct.
    """
    def play(order):
        eng, remaining, val, cost = Engine(), global_budget, 0.0, 0
        for t in order:
            if remaining <= 0:
                break
            ok, c, _ = eng.solve(t, min(per_task_cap, remaining), use_memory=False, reconfigure=False)
            remaining -= c; cost += c
            if ok:
                val += t.value
        return val, cost
    naive = play(list(tasks))                                        # original order
    value_only = play(sorted(tasks, key=lambda t: t.value, reverse=True))
    routed = play(sorted(tasks, key=lambda t: t.density, reverse=True))  # value/cost
    return naive, value_only, routed


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
def _demo(emit_path: str | None = None):
    tasks = task_suite()
    BUD = 800

    t0 = time.perf_counter()
    _, cold_cost, cold_solved, cold_rec = run_suite(
        tasks, use_memory=False, reconfigure=False, per_task_budget=BUD)
    warm_eng, warm_cost, warm_solved, warm_rec = run_suite(
        tasks, use_memory=True, reconfigure=False, per_task_budget=BUD)
    # reconfiguration probe (kept honest: measured, not assumed beneficial)
    _, recfg_cost, _, _ = run_suite(tasks, use_memory=True, reconfigure=True, per_task_budget=BUD)
    dt = time.perf_counter() - t0

    print("== Closed-loop multiplier (REAL executions) ==")
    print(f"cold (no memory)        : solved {cold_solved}/{len(tasks)}  cost={cold_cost} candidates")
    print(f"warm (memory)           : solved {warm_solved}/{len(tasks)}  cost={warm_cost} candidates")
    if warm_solved == cold_solved and warm_cost > 0:
        print(f"MULTIPLIER (cost_cold / cost_warm at equal correctness) = x{cold_cost / warm_cost:.2f}")
    else:
        print("correctness differs -> multiplier not comparable (inspect tasks)")
    print(f"memory compression: {warm_eng.solved_count} solves stored as "
          f"{sum(len(v) for v in warm_eng.memory.values())} patterns "
          f"(ratio {warm_eng.compression_ratio():.2f} solves/pattern, bounded)")
    print(f"reconfiguration probe   : memory+reconfig cost={recfg_cost} "
          f"(x{cold_cost / recfg_cost:.2f}) -> "
          f"{'helps' if recfg_cost < warm_cost else 'does NOT help here; left OFF'}")
    print(f"wall-clock: {dt*1000:.0f} ms")

    print("\n== Value routing under a hard global budget ==")
    gb = int(cold_cost * 0.35)
    (nv, nc), (ev, ec), (rv, rc) = value_routing(tasks, global_budget=gb, per_task_cap=600)
    print(f"global budget = {gb} candidates")
    print(f"naive order       : value captured {nv:.0f}  (cost {nc})")
    print(f"by value only     : value captured {ev:.0f}  (cost {ec})   <- the trap")
    print(f"by value/cost     : value captured {rv:.0f}  (cost {rc})   <- correct routing")
    if nv:
        print(f"VALUE GAIN of density-routing vs naive = x{rv/nv:.2f}; vs value-only = x{rv/max(ev,1e-9):.2f}")

    if emit_path:
        _emit(emit_path, cold_rec, warm_rec)
        print(f"\nwrote CEE records -> {emit_path} (configs: cold, warm)")


def _emit(path, cold_rec, warm_rec):
    """Write CEE-scorable TaskRun records. token-proxy = candidates*8 (real compute)."""
    def rec(t: Task, ok: bool, cost: int, cfg: str):
        return {
            "task_id": t.tid, "config": cfg,
            "tokens_in": cost * 6, "tokens_out": cost * 2,
            "useful_decisions": 1 if ok else 0, "total_decisions": 1,
            "actions_succeeded": 1 if ok else 0, "actions_attempted": 1,
            "solution_quality": 1.0 if ok else 0.0,
            "complexity": min(1.0, t.value / 5.0),
            "human_baseline_s": 60 * t.value, "oversight_s": 0,
        }
    with open(path, "w") as f:
        for t, ok, c in cold_rec:
            f.write(json.dumps(rec(t, ok, c, "cold")) + "\n")
        for t, ok, c in warm_rec:
            f.write(json.dumps(rec(t, ok, c, "warm")) + "\n")


def _selftest() -> int:
    # real execution sanity
    assert run_program((("mul2",), None), [1, 2, 3]) == [2, 4, 6]
    assert run_program((("feven",), "sum"), [1, 2, 3, 4]) == 6
    assert run_program((("srtd", "take2"), None), [5, 1, 4]) == [5, 4]
    # closed loop solves and reuses: second instance of a family must be cheaper warm
    eng = Engine()
    tasks = task_suite()
    fam = [t for t in tasks if t.family == "scale"]
    ok1, c1, p1 = eng.solve(fam[0], 800, use_memory=True, reconfigure=False)
    ok2, c2, p2 = eng.solve(fam[1], 800, use_memory=True, reconfigure=False)
    assert ok1 and ok2 and p1 == p2, (ok1, ok2, p1, p2)
    assert c2 < c1, f"memory did not reduce cost: {c1} -> {c2}"
    # whole-suite multiplier must be > 1 (memory cheaper at equal correctness)
    _, cold, cs, _ = run_suite(tasks, use_memory=False, reconfigure=False, per_task_budget=800)
    _, warm, ws, _ = run_suite(tasks, use_memory=True, reconfigure=False, per_task_budget=800)
    assert cs == ws and warm < cold, (cs, ws, cold, warm)
    print(f"selftest OK: loop executes, reuses (cost {c1}->{c2}), suite multiplier x{cold/warm:.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Token Multiplication Engine (closed loop)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", metavar="PATH", help="write CEE-scorable records")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.demo or a.emit:
        _demo(a.emit)
        return 0
    ap.error("use --demo, --selftest, or --emit PATH")


if __name__ == "__main__":
    sys.exit(main())
