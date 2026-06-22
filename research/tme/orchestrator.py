"""Hierarchical Orchestrator — the missing core.

Turns a MINIMAL intention into a working, self-improving pipeline. This is the
layer the critique correctly identified as absent: not "more actions per token"
(a false fixed equivalence) but *variable amplification by architecture* —
abstraction + decomposition + automation + memory, measured.

Maps to the five levels of cognitive leverage:
  L1 expand intention   : parse a compact intention -> structured objective
  L2 decompose          : objective -> sub-tasks (one verb can fan out to many)
  L3 generate           : each sub-task -> a synthesized program (via TME search)
  L4 automate+test      : execute on real data, validate end-to-end on held-out
  L5 improvement loop    : TME memory/reconfig carry across the cascade & across runs

Every number is REAL (synthesized programs really run on real data) and VARIABLE:
amplification depends on the architecture and the intention, never a fixed ratio.
On a structureless intention it collapses toward 1 — the correct, non-magical
behavior.

    python3 orchestrator.py --selftest
    python3 orchestrator.py --demo
"""

from __future__ import annotations

import argparse
import sys

from tme import Engine, Task, run_program, TERMINALS  # closed-loop substrate

# --------------------------------------------------------------------------- #
# Intention vocabulary. A compact verb expands to a reference program; the
# orchestrator does NOT hand this program to the solver — it derives I/O examples
# from it and makes TME rediscover an executable program by real search.
# --------------------------------------------------------------------------- #
TRANSFORMS = {
    "double": (("mul2",), None),
    "inc": (("add1",), None),
    "evens": (("feven",), None),
    "odds": (("fodd",), None),
    "runsum": (("psum",), None),
    "rev": (("rev",), None),
    "sortd": (("srtd",), None),
    "dedup": (("dedup",), None),
}
REPORTS = {name: ((), name) for name in TERMINALS}          # scalar terminals
GROUPS = {"summary": ["sum", "max", "min"]}                 # one token -> 3 sub-tasks


def _transform_ref(verb: str):
    if verb.startswith("top:"):
        k = int(verb.split(":")[1])
        return (("srtd", f"take{k}"), None)
    return TRANSFORMS[verb]


def parse(intention: str):
    """L1: compact intention -> (stages, reports). Token count = atoms in the spec."""
    head, _, tail = intention.partition(">")
    stages = [s.strip() for s in head.split("|") if s.strip()]
    reports: list[str] = []
    for r in (x.strip() for x in tail.split(",") if x.strip()):
        reports.extend(GROUPS.get(r, [r]))                  # L2 fan-out
    tokens = len([s for s in head.split("|") if s.strip()]) + \
        len([x for x in tail.split(",") if x.strip()])      # atoms typed by the user
    return stages, reports, max(tokens, 1)


def _ops(prog) -> int:
    return len(prog[0]) + (1 if prog[1] is not None else 0)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class Orchestrator:
    def __init__(self, budget: int = 800):
        self.eng = Engine()        # shared across the whole cascade => L5 reuse
        self.budget = budget

    def run_intention(self, intention: str, data: list[list[int]], holdout: list[list[int]]):
        stages, reports, tokens = parse(intention)
        synth: list = []           # synthesized stage programs (the generated "system")
        build_cost = 0             # real search work (candidate evaluations)
        exec_ops = 0               # real primitive-op executions on data (leaf actions)
        running = [list(r) for r in data]

        # L3+L4: synthesize and execute each transform stage in cascade
        for verb in stages:
            ref = _transform_ref(verb)
            pairs = [(r, run_program(ref, r)) for r in running]
            task = Task(f"stage:{verb}", verb, 1.0, pairs)
            ok, cost, prog = self.eng.solve(task, self.budget, use_memory=True, reconfigure=False)
            build_cost += cost
            if not ok:
                return _Result(intention, tokens, synth, build_cost, exec_ops, 0.0, "unsolved:" + verb)
            synth.append(("stage", verb, prog))
            exec_ops += _ops(prog) * len(running)
            running = [run_program(prog, r) for r in running]    # real cascade

        # L2/L3 report fan-out on the final data
        report_progs = []
        for name in reports:
            ref = REPORTS[name]
            pairs = [(r, run_program(ref, r)) for r in running]
            task = Task(f"report:{name}", name, 1.0, pairs)
            ok, cost, prog = self.eng.solve(task, self.budget, use_memory=True, reconfigure=False)
            build_cost += cost
            if not ok:
                return _Result(intention, tokens, synth, build_cost, exec_ops, 0.0, "unsolved:" + name)
            report_progs.append(prog)
            synth.append(("report", name, prog))
            exec_ops += _ops(prog) * len(running)

        # L4 validation on HELD-OUT data: does the generated system generalize?
        q = self._validate(stages, reports, report_progs,
                           [p for k, _, p in synth if k == "stage"], holdout)
        return _Result(intention, tokens, synth, build_cost, exec_ops, q, "ok")

    def _validate(self, stages, reports, report_progs, stage_progs, holdout) -> float:
        ref_run = [list(r) for r in holdout]
        syn_run = [list(r) for r in holdout]
        for verb, prog in zip(stages, stage_progs):
            ref = _transform_ref(verb)
            ref_run = [run_program(ref, r) for r in ref_run]
            syn_run = [run_program(prog, r) for r in syn_run]
        checks, ok = 0, 0
        for r_ref, r_syn in zip(ref_run, syn_run):
            checks += 1; ok += int(r_ref == r_syn)
        for name, prog in zip(reports, report_progs):
            ref = REPORTS[name]
            for rr, sr in zip(ref_run, syn_run):
                checks += 1; ok += int(run_program(ref, rr) == run_program(prog, sr))
        return ok / checks if checks else 0.0


class _Result:
    def __init__(self, intention, tokens, synth, build_cost, exec_ops, quality, status):
        self.intention = intention
        self.tokens = tokens
        self.subprograms = len(synth)
        self.synth_ops = sum(_ops(p) for _, _, p in synth)
        self.build_cost = build_cost
        self.exec_ops = exec_ops
        self.quality = quality
        self.status = status

    @property
    def decomposition_factor(self):   # sub-programs generated per intention atom
        return self.subprograms / self.tokens

    @property
    def abstraction_leverage(self):   # concrete ops written by the system per atom
        return self.synth_ops / self.tokens

    @property
    def execution_amplification(self):  # real leaf actions executed per atom
        return self.exec_ops / self.tokens


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
DATA = [[3, 1, 2, 2], [5, 4], [6, 1, 3], [2, 2, 8, 4]]
HELD = [[7, 1, 5], [2, 4, 6, 8], [9, 9, 2]]
INTENTIONS = [
    "double | evens | runsum > summary",
    "top:2 | rev > sum,max",
    "inc | dedup | sortd > summary",
]


def _demo():
    print("== Hierarchical orchestration: minimal intention -> running system ==\n")
    orch = Orchestrator()
    tot_tokens = tot_exec = tot_build = 0
    for intent in INTENTIONS:
        r = orch.run_intention(intent, DATA, HELD)
        tot_tokens += r.tokens; tot_exec += r.exec_ops; tot_build += r.build_cost
        print(f"intention: {r.intention!r}  [{r.status}]")
        print(f"  {r.tokens} atoms -> {r.subprograms} synthesized sub-programs "
              f"(decomposition x{r.decomposition_factor:.1f})")
        print(f"  abstraction leverage: x{r.abstraction_leverage:.1f} concrete ops / atom")
        print(f"  execution amplification: x{r.execution_amplification:.1f} real leaf actions / atom")
        print(f"  build cost: {r.build_cost} candidates | held-out validation Q={r.quality:.2f}\n")

    # L5: the improvement loop. Reuse across DIFFERENT intentions is modest when
    # they share little structure (correct: memory cannot multiply the unseen).
    # The loop shows its teeth on REPEATED work: a second pass over the same
    # intentions should collapse, because the system now remembers how.
    cold = sum(_one_cold(i) for i in INTENTIONS)
    pass2 = sum(orch.run_intention(i, DATA, HELD).build_cost for i in INTENTIONS)
    print("== L5 improvement loop (the system gets cheaper as it learns) ==")
    print(f"pass 1 cold (no memory)                 : {cold} candidates")
    print(f"pass 1 warm (memory, novel intentions)  : {tot_build} candidates  "
          f"(x{cold / tot_build:.2f} — small: little shared structure)")
    print(f"pass 2 warm (same intentions, learned)  : {pass2} candidates  "
          f"(x{cold / max(pass2,1):.1f} — repeated work collapses)")
    print(f"\naggregate (pass 1): {tot_tokens} intention atoms -> {tot_exec} real leaf "
          f"actions (x{tot_exec / tot_tokens:.1f}); amplification is VARIABLE, not a fixed ratio.")


def _one_cold(intent: str) -> int:
    return Orchestrator().run_intention(intent, DATA, HELD).build_cost


def _selftest() -> int:
    orch = Orchestrator()
    r = orch.run_intention("double | evens | runsum > summary", DATA, HELD)
    assert r.status == "ok", r.status
    assert r.quality == 1.0, r.quality                       # generalizes to held-out
    assert r.subprograms == 6, r.subprograms                 # 3 stages + summary(3)
    assert r.decomposition_factor > 1.0                      # 6 sub-programs / 4 atoms
    assert r.execution_amplification > 1.0
    # structureless single-atom intention must NOT magically amplify much
    r2 = Orchestrator().run_intention("rev", [[1, 2]], [[3, 4]])
    assert r2.subprograms == 1 and r2.tokens == 1
    # cascade reuse: warm batch <= cold batch
    warm = Orchestrator()
    wb = sum(warm.run_intention(i, DATA, HELD).build_cost for i in INTENTIONS)
    cb = sum(_one_cold(i) for i in INTENTIONS)
    assert wb <= cb, (wb, cb)
    print(f"selftest OK: intention->system validated (Q=1.0), decomposition x{r.decomposition_factor:.1f}, "
          f"cascade reuse {cb}->{wb}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hierarchical Orchestrator over TME")
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
