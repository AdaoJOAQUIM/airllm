"""Self-extending abstraction (library learning) — single-level AND hierarchical.

Two regimes, both deterministic, both held-out validated:

* paradigm_report()  : single-level proof that representation change beats caching
  (TRANSFER + MDL compression + effective DEPTH, with an honest guardrail).

* curriculum_report(): the abstraction PUSHED FURTHER -- HIERARCHICAL macros (macros
  built from macros). The engine bootstraps a DEPTH LADDER: each learned level lets
  it reach one base-op deeper while keeping SEARCH depth <= MAXD. Effective base
  depth climbs 2 -> 3 -> 4 -> 5 ... and the "LoC per written symbol" ratio climbs
  with it.

Claude Code is the PROPOSER. In real use Claude supplies the compact intentions /
abstraction hints; this kernel deterministically validates, executes, compresses and
persists. So "1 Claude token ~ several actions / LoC" is a MEASURED, VARIABLE ratio
(base ops emitted+run per written symbol), not a fixed equivalence. It grows as the
library deepens and collapses to ~1 with no reusable structure.

    python3 abstraction.py --selftest
    python3 abstraction.py --demo
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from tme import (Engine, Task, run_program, LEARNED, MAXD,
                 base_length, symbol_length)

# Negatives break coincidences (with only positives, psum is monotonic so
# rev(psum)==srtd(psum) and distinct programs collapse to one function).
INPUTS = [[1, -2, 3], [4, 5, -6], [2, 7, -1, 8], [-3, 4, 9], [6, 1, -4, 2], [2, -2, 4, -6, 8]]
HELD = [[5, -2, 7, -8], [-1, 6, 3, -4], [10, -5, 2, 2]]

REF: dict[str, tuple] = {}     # tid -> reference program, for held-out validation


def task_from_ops(tid, base_seq, term=None, inputs=INPUTS, value=1.0) -> Task:
    ref = (tuple(base_seq), term)
    REF[tid] = ref
    return Task(tid, tid, value, [(xs, run_program(ref, xs)) for xs in inputs])


def solve_all(engine, tasks, budget=1500):
    """Solve each task, COUNT it only if the program GENERALIZES to held-out inputs
    (vs reference). Blocks spurious fits -- the honest bar for 'solved'."""
    progs, costs = {}, {}
    for t in tasks:
        ok, c, p = engine.solve(t, budget, use_memory=True, reconfigure=False)
        costs[t.tid] = c
        if ok and all(run_program(p, h) == run_program(REF[t.tid], h) for h in HELD):
            progs[t.tid] = p
    return progs, costs


# --------------------------------------------------------------------------- #
# MDL library learner (single-level and symbol-level for hierarchy)
# --------------------------------------------------------------------------- #
def _expand(names):
    out = []
    for x in names:
        out.extend(LEARNED[x] if x in LEARNED else (x,))
    return tuple(out)


def compress_names(names, macros):
    """Re-express a sequence with the library (greedy longest base-expansion match),
    yielding current symbols (macro names allowed). Enables hierarchical mining."""
    by_len = sorted(macros, key=lambda m: base_length((m[1], None)), reverse=True)
    base = list(_expand(names))
    out = []
    i = 0
    while i < len(base):
        for name, _seq in by_len:
            exp = list(_expand((name,)))
            if base[i:i + len(exp)] == exp:
                out.append(name); i += len(exp); break
        else:
            out.append(base[i]); i += 1
    return tuple(out)


def _mine(seqs, min_count=2, max_len=3):
    """Pick the fragment that most reduces description length.
    score = count*(len-1) - len  (savings minus one-time library cost); need > 0."""
    counts: Counter = Counter()
    for s in seqs:
        for L in range(2, max_len + 1):
            for i in range(len(s) - L + 1):
                counts[tuple(s[i:i + L])] += 1
    best, best_score = None, 0
    for frag, cnt in counts.items():
        if cnt < min_count:
            continue
        score = cnt * (len(frag) - 1) - len(frag)
        if score > best_score:
            best, best_score = frag, score
    return best, best_score


def compress_len(names, macros) -> int:
    return len(compress_names(names, macros))


# --------------------------------------------------------------------------- #
# Single-level paradigm proof (kept): transfer + MDL + depth + guardrail
# --------------------------------------------------------------------------- #
def _training():
    return [task_from_ops("tr1", ("feven", "psum")),
            task_from_ops("tr2", ("feven", "psum"), inputs=[[2, 3, 4], [5, -6, 7, 8]]),
            task_from_ops("tr3", ("feven", "psum"), inputs=[[8, 1, -2], [4, 4, -5]]),
            task_from_ops("tr4", ("mul2",)), task_from_ops("tr5", ("rev",))]


def _novel():
    return [task_from_ops("nv1", ("feven", "psum", "rev")),
            task_from_ops("nv2", ("feven", "psum", "srtd"))]


def _guard():     # depth-3, irreducible (reverse of 2x+2), unrelated to the macro
    return [task_from_ops("gr1", ("add1", "mul2", "rev"))]


def learn(engine, train, rounds=2):
    macros = []
    for _ in range(rounds):
        progs, _ = solve_all(engine, train)
        sym = [compress_names(p[0], macros) for p in progs.values()]
        frag, score = _mine(sym)
        if not frag or score <= 0 or any(seq == frag for _, seq in macros):
            break
        name = f"m{len(macros)}"; engine.add_macro(name, frag); macros.append((name, frag))
    return macros


def paradigm_report(engine=None, verbose=True):
    LEARNED.clear()
    train, novel, guard = _training(), _novel(), _guard()
    eng = engine or Engine()
    base_progs, _ = solve_all(eng, train)
    mdl_before = sum(base_length(p) for p in base_progs.values())
    nov_before, _ = solve_all(Engine(), novel)
    grd_before, _ = solve_all(Engine(), guard)
    macros = learn(eng, train)
    mdl_after = sum(compress_len(p[0], macros) + (1 if p[1] is not None else 0)
                    for p in base_progs.values()) + sum(len(s) for _, s in macros)
    nov_after, _ = solve_all(eng, novel)
    grd_after, _ = solve_all(eng, guard)
    if verbose:
        print(f"learned macros: {macros}\n")
        print(f"1) TRANSFER (novel depth-3 > base MAXD={MAXD}): "
              f"{len(nov_before)}/{len(novel)} -> {len(nov_after)}/{len(novel)}")
        print(f"2) MDL COMPRESSION (corpus): {mdl_before} -> {mdl_after} symbols")
        print(f"3) EFFECTIVE DEPTH: " + ", ".join(
            f"{t}={p[0]}(d{symbol_length(p)}==base d{base_length(p)})"
            for t, p in sorted(nov_after.items())))
        print(f"GUARDRAIL (unrelated deep): {len(grd_before)} -> {len(grd_after)} (stays 0)")
    return {"macros": macros, "novel_before": len(nov_before), "novel_after": len(nov_after),
            "novel_total": len(novel), "mdl_before": mdl_before, "mdl_after": mdl_after,
            "guard_before": len(grd_before), "guard_after": len(grd_after),
            "novel_progs": nov_after}


# --------------------------------------------------------------------------- #
# Hierarchical curriculum: bootstrap a DEPTH LADDER (abstraction pushed further)
# --------------------------------------------------------------------------- #
# Repeated mul2 => powers of two. No power beyond x2/x3 is a single base op, and no
# composition collapses to fewer ops, so CHAIN[:L] genuinely needs L base ops. This
# avoids the algebraic collapses (srtd.rev==srt, dedup==id on data) that fake depth.
CHAIN = ("mul2", "mul2", "mul2", "mul2", "mul2")
_VAR_INPUTS = [INPUTS, [[2, 3, -4], [5, -6, 7, 8]], [[8, 1, -2, 3], [4, -4, 5, -3]]]


def _stage(level):   # ref = CHAIN[:level+1]; three tasks (same ref, varied inputs)
    return [task_from_ops(f"s{level}_{i}", CHAIN[:level + 1], None, inp)
            for i, inp in enumerate(_VAR_INPUTS)]


def reach(engine):
    """Max base depth L (>=2) for which CHAIN[:L] is solvable at search depth <= MAXD,
    held-out validated."""
    last = 1
    for L in range(2, len(CHAIN) + 1):
        t = task_from_ops(f"reach{L}", CHAIN[:L])
        progs, _ = solve_all(engine, [t])
        if not progs:
            break
        last = L
    return last


def learn_curriculum(engine, levels=(1, 2, 3)):
    """Each stage learns one HIERARCHICAL macro (macros of macros) -> +1 base depth.
    A single accumulating library is kept so later macros can reference earlier ones.
    Returns (macros, depth_ladder)."""
    macros = []
    ladder = [reach(engine)]                      # before learning (expect 2)
    for lv in levels:
        progs, _ = solve_all(engine, _stage(lv))
        sym = [compress_names(p[0], macros) for p in progs.values()]
        frag, score = _mine(sym)
        if frag and score > 0 and not any(seq == frag for _, seq in macros):
            name = f"m{len(macros)}"
            engine.add_macro(name, frag); macros.append((name, frag))
        ladder.append(reach(engine))
    return macros, ladder


def curriculum_report(engine=None, verbose=True):
    LEARNED.clear()
    eng = engine or Engine()
    macros, ladder = learn_curriculum(eng)
    # novel deep task (depth 5): solvable only via the stacked library
    novel = task_from_ops("deep", CHAIN[:5])
    nprog, _ = solve_all(eng, [novel])
    guard, _ = solve_all(eng, _guard())          # unrelated -> must stay unsolved
    deep = nprog.get("deep")
    # Claude-as-proposer amplification: base ops (LoC) emitted+run per written symbol
    amp = None
    if deep:
        loc_per_symbol = base_length(deep) / symbol_length(deep)
        actions = base_length(deep) * len(INPUTS)
        amp = (symbol_length(deep), base_length(deep), loc_per_symbol, actions)
    if verbose:
        print(f"hierarchical macros (macros of macros): {macros}\n")
        print(f"DEPTH LADDER (search stays <= MAXD={MAXD}): "
              + " -> ".join(map(str, ladder)) + "  (effective base depth)")
        if deep:
            s, b, r, a = amp
            print(f"\nnovel DEEP task CHAIN[:5] solved as {deep[0]}  "
                  f"(written depth {s}, base depth {b})")
            print(f"CLAUDE-AS-PROPOSER amplification: {s} written symbols -> {b} LoC "
                  f"(x{r:.1f} LoC/symbol) -> {a} real actions on {len(INPUTS)} inputs")
            print("  (variable: deeper library => higher LoC-per-symbol; ~1 with no reuse)")
        print(f"GUARDRAIL (unrelated deep): solved {len(guard)} (stays 0 -- not magic)")
    return {"macros": macros, "ladder": ladder, "deep": deep,
            "guard": len(guard), "amp": amp}


def _demo():
    print("== Self-extending abstraction ==\n--- single-level proof ---")
    paradigm_report()
    print("\n--- hierarchical: depth ladder + Claude-as-proposer amplification ---")
    curriculum_report()


def _selftest() -> int:
    r = paradigm_report(verbose=False)
    assert r["macros"] and r["macros"][0][1] == ("feven", "psum"), r["macros"]
    assert r["novel_before"] == 0 and r["novel_after"] == r["novel_total"], r
    assert r["mdl_after"] < r["mdl_before"], r
    assert r["guard_before"] == 0 and r["guard_after"] == 0, r

    c = curriculum_report(verbose=False)
    assert len(c["macros"]) >= 2, c["macros"]
    assert c["macros"][1][1][0] == "m0", c["macros"]        # hierarchy: m1 references m0
    assert c["ladder"][0] == 2, c["ladder"]                 # base reaches depth 2
    assert c["ladder"][-1] >= 4, c["ladder"]                # ladder climbed
    assert c["deep"] is not None, "deep novel task not solved by stacked library"
    assert symbol_length(c["deep"]) <= MAXD < base_length(c["deep"]), c["deep"]
    assert c["guard"] == 0, c
    assert c["amp"][2] > 1.0, c["amp"]                       # LoC per written symbol > 1
    print(f"selftest OK: single-level transfer 0->{r['novel_after']}/{r['novel_total']}, "
          f"MDL {r['mdl_before']}->{r['mdl_after']}; hierarchy ladder {c['ladder']}, "
          f"deep solved {c['deep'][0]} (x{c['amp'][2]:.1f} LoC/symbol)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-extending abstraction (library learning)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.demo:
        _demo(); return 0
    ap.error("use --demo or --selftest")


if __name__ == "__main__":
    sys.exit(main())
