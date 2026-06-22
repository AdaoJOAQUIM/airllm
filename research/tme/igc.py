"""Intent -> Execution Graph Compiler (IGC).

The piece between an *optimizer* (TME) and a *cognitive orchestrator*: it compiles a
minimal intention into a CAUSAL ACTION GRAPH (DAG) and allocates compute by impact,
rather than running a linear pipeline.

The real multiplier is NOT "actions per token" but DEPTH OF COMPILATION:
  - shared sub-results execute ONCE and feed many consumers (a list cannot do this)
  - dependency-aware pruning skips/compresses low-impact nodes under a budget

Graph:  node = action,  edge = dependency,  weight = (value, cost).

Amplification Factor = useful executed graph nodes / intention atoms
                       (variable: depends on structure, memory, and budget).

Intention DSL (statements separated by ';'):
  name = op(arg)        transform node      (arg = 'input' or another name)
  name = stat(arg)      report/goal node    (stat in sum/max/min/len), value 1
  ... append '@v' to set a goal's value, e.g.  g = max(x)@5
A name reused as an arg creates a SHARED node (fan-out) -> that is the whole point.

    python3 igc.py --selftest
    python3 igc.py --demo
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from tme import Engine, Task, run_program
from orchestrator import TRANSFORMS, REPORTS, DATA, HELD

# transform vocabulary (params folded into fixed names so parsing stays a one-liner)
TREF = dict(TRANSFORMS)
TREF["top2"] = (("srtd", "take2"), None)
TREF["top3"] = (("srtd", "take3"), None)


# --------------------------------------------------------------------------- #
# Compilation: intention text -> DAG (deduped transform nodes + goal nodes)
# --------------------------------------------------------------------------- #
@dataclass
class Graph:
    tokens: int
    transforms: dict[str, tuple[str, str]]      # key -> (op, parent_key); deduped
    goals: list[tuple[str, str, str, float]]    # (name, stat, parent_key, value)

    def chain(self, key: str) -> list[str]:
        """Transform-node keys from input up to and including `key`."""
        if key == "input":
            return []
        op, parent = self.transforms[key]
        return self.chain(parent) + [key]

    def depth(self) -> int:
        return max((len(self.chain(pk)) for _, _, pk, _ in self.goals), default=0)


def compile_intention(intention: str) -> Graph:
    stmts = [s.strip() for s in intention.split(";") if s.strip()]
    env: dict[str, str] = {}            # name -> structural key
    transforms: dict[str, tuple[str, str]] = {}
    goals: list[tuple[str, str, str, float]] = []
    for st in stmts:
        name, rhs = (x.strip() for x in st.split("=", 1))
        value = 1.0
        if "@" in rhs:
            rhs, v = rhs.rsplit("@", 1)
            value = float(v)
        func = rhs[: rhs.index("(")].strip()
        arg = rhs[rhs.index("(") + 1 : rhs.rindex(")")].strip()
        parent_key = "input" if arg == "input" else env[arg]
        if func in REPORTS:
            goals.append((name, func, parent_key, value))
            env[name] = parent_key
        else:
            key = f"{func}({parent_key})"      # structural key -> automatic dedup
            transforms[key] = (func, parent_key)
            env[name] = key
    return Graph(len(stmts), transforms, goals)


# --------------------------------------------------------------------------- #
# Scheduling: allocate a compute budget by value-density over the graph
# --------------------------------------------------------------------------- #
def schedule(g: Graph, budget: int, *, share: bool):
    """Greedy by value/cost. `share=True` reuses overlapping ancestors (the graph
    advantage); `share=False` is the linear baseline that recomputes per goal."""
    required: set[str] = set()
    chosen: list = []
    cost = 0
    order = sorted(g.goals, key=lambda gl: gl[3] / (len(g.chain(gl[2])) + 1), reverse=True)
    for goal in order:
        ch = g.chain(goal[2])
        marginal = (len(set(ch) - required) if share else len(ch)) + 1   # +1 = the report
        if cost + marginal <= budget:
            chosen.append(goal)
            if share:
                required |= set(ch)
            cost += marginal
    return chosen, required, cost


# --------------------------------------------------------------------------- #
# Execution: synthesize + run required nodes once, on real data
# --------------------------------------------------------------------------- #
class Compiler:
    def __init__(self, budget_synth: int = 800):
        self.eng = Engine()                 # directed memory shared across nodes
        self.budget_synth = budget_synth
        self.synth_cost = 0
        self.node_execs = 0

    def _node_prog(self, op_name: str, parent_data: list[list[int]]):
        ref = TREF[op_name] if op_name in TREF else ((), op_name)   # transform or report
        pairs = [(r, run_program(ref, r)) for r in parent_data]
        ok, c, prog = self.eng.solve(Task(op_name, op_name, 1.0, pairs),
                                     self.budget_synth, use_memory=True, reconfigure=False)
        self.synth_cost += c
        return ok, prog

    def execute(self, g: Graph, required: set[str], chosen: list, data):
        """Run required transform nodes once (shared), then committed report nodes."""
        cache: dict[str, list] = {"input": [list(r) for r in data]}
        # transform nodes in dependency order (by chain length)
        for key in sorted(required, key=lambda k: len(g.chain(k))):
            op, parent = g.transforms[key]
            ok, prog = self._node_prog(op, cache[parent])
            if not ok:
                return None
            cache[key] = [run_program(prog, r) for r in cache[parent]]
            self.node_execs += len(cache[parent])
            cache[key + "#prog"] = prog
        results = {}
        for name, stat, pk, _ in chosen:
            ok, prog = self._node_prog(stat, cache[pk])
            if not ok:
                return None
            results[name] = (stat, pk, prog)
            self.node_execs += len(cache[pk])
        return cache, results

    def validate(self, g: Graph, cache, results, holdout) -> float:
        """Held-out check: does the compiled+synthesized graph generalize?"""
        # rebuild required transform results on held-out via synthesized progs
        h: dict[str, list] = {"input": [list(r) for r in holdout]}
        href: dict[str, list] = {"input": [list(r) for r in holdout]}
        for key in sorted([k for k in cache if k != "input" and not k.endswith("#prog")],
                          key=lambda k: len(g.chain(k))):
            op, parent = g.transforms[key]
            h[key] = [run_program(cache[key + "#prog"], r) for r in h[parent]]
            href[key] = [run_program(TREF[op], r) for r in href[parent]]
        ok = checks = 0
        for name, (stat, pk, prog) in results.items():
            ref = ((), stat)
            for rs, rr in zip(h[pk], href[pk]):
                checks += 1
                ok += int(run_program(prog, rs) == run_program(ref, rr))
        return ok / checks if checks else 0.0


# --------------------------------------------------------------------------- #
# Metrics report
# --------------------------------------------------------------------------- #
def report(intention: str, budget: int):
    g = compile_intention(intention)

    # graph vs linear scheduling under the same hard budget
    g_chosen, g_req, g_cost = schedule(g, budget, share=True)
    l_chosen, _, l_cost = schedule(g, budget, share=False)
    g_val = sum(c[3] for c in g_chosen)
    l_val = sum(c[3] for c in l_chosen)

    # real execution of the graph-scheduled plan
    comp = Compiler()
    out = comp.execute(g, g_req, g_chosen, DATA)
    q = comp.validate(g, out[0], out[1], HELD) if out else 0.0

    naive_execs = sum(len(g.chain(c[2])) for c in g_chosen)   # recompute-per-goal
    shared_nodes = len(g_req)
    print(f"intention ({g.tokens} atoms): {intention!r}")
    print(f"  compiled graph: {len(g.transforms)} transform nodes, {len(g.goals)} goals, "
          f"depth {g.depth()}")
    print(f"  shared-node reuse: {naive_execs} recompute-execs -> {shared_nodes} distinct "
          f"(x{naive_execs / max(shared_nodes,1):.2f})")
    print(f"  execution amplification: {comp.node_execs} real leaf actions / {g.tokens} "
          f"atoms = x{comp.node_execs / g.tokens:.2f}")
    print(f"  budget={budget}: graph captures value {g_val:.0f} (cost {g_cost}) vs "
          f"linear {l_val:.0f} (cost {l_cost})  -> x{g_val / max(l_val,1e-9):.2f}")
    print(f"  synth cost {comp.synth_cost} candidates | held-out Q={q:.2f}\n")
    return g, g_val, l_val, q


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
DIAMOND = "x=double(input); e=evens(x); s=runsum(e); a=sum(s)@2; b=max(x)@3; c=min(e)@1"
LINEAR = "x=inc(input); y=runsum(x); z=rev(y); g=sum(z)@2"


def _demo():
    print("== Intent -> Execution Graph Compiler ==")
    print("Depth of compilation, not actions-per-token. Amplification is structural.\n")
    report(DIAMOND, budget=5)      # tight budget -> graph sharing wins value
    report(LINEAR, budget=8)       # no sharing -> amplification collapses toward 1
    print("Reading: the diamond shares `x`/`e` across goals (graph reuse > 1, more")
    print("value captured per budget than the linear recompute-per-goal baseline).")
    print("The linear intention has nothing to share -> reuse ~1. Structure-dependent,")
    print("exactly as a real compiler should be -- never a fixed equivalence.")


def _selftest() -> int:
    g = compile_intention(DIAMOND)
    assert len(g.transforms) == 3, g.transforms          # double, evens, runsum (deduped)
    assert len(g.goals) == 3
    assert g.depth() == 3
    # shared scheduling must be cheaper than per-goal recompute for the same goals
    chosen, req, cost = schedule(g, budget=99, share=True)
    naive = sum(len(g.chain(c[2])) for c in chosen)
    assert naive > len(req), (naive, len(req))           # real graph reuse
    # real execution generalizes on held-out
    comp = Compiler()
    out = comp.execute(g, req, chosen, DATA)
    assert out is not None
    q = comp.validate(g, out[0], out[1], HELD)
    assert q == 1.0, q
    # under a tight budget, graph captures >= linear value
    g_chosen, _, _ = schedule(g, 5, share=True)
    l_chosen, _, _ = schedule(g, 5, share=False)
    assert sum(c[3] for c in g_chosen) >= sum(c[3] for c in l_chosen)
    print(f"selftest OK: graph reuse {naive}->{len(req)} nodes, held-out Q={q:.2f}, "
          f"graph value >= linear under budget")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Intent -> Execution Graph Compiler")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.demo:
        _demo()
        return 0
    ap.error("use --demo or --selftest")


if __name__ == "__main__":
    sys.exit(main())
