"""Self-extending abstraction (library learning) — the paradigm step.

v1 (engine.py) MEMOIZES: it caches exact solutions. This module makes the engine
CHANGE ITS OWN REPRESENTATION: it mines its solved programs, compresses the most
recurring structure into NEW primitives (macros), grows its DSL, and thereby solves
NOVEL tasks it could not solve before -- at equal compute. That is the difference
between a cache and learning.

The proof it is NOT a cache (all measured by --demo, all required):
  1. Transfer : novel held-out tasks beyond the base DSL's depth go 0 -> >0 solved.
  2. MDL      : description length of the solved corpus drops after learning.
  3. Depth    : a depth-3 base solution is reached at depth-2 cost via a macro.
Guardrail: a novel deep task UNRELATED to the learned macro stays unsolved (honest).

    python3 abstraction.py --selftest
    python3 abstraction.py --demo
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from tme import (Engine, Task, run_program, LEARNED, MAXD,
                 base_length, symbol_length)

# --------------------------------------------------------------------------- #
# Task construction from a reference program (so we have ground truth + held-out)
# --------------------------------------------------------------------------- #
# Negatives matter: with only positives, psum is monotonic so rev(psum)==srtd(psum)
# and several "different" programs collapse to the same function. Negatives break
# those coincidences so distinct programs are genuinely distinguishable.
INPUTS = [[1, -2, 3], [4, 5, -6], [2, 7, -1, 8], [-3, 4, 9], [6, 1, -4, 2], [2, -2, 4, -6, 8]]
HELD = [[5, -2, 7, -8], [-1, 6, 3, -4], [10, -5, 2, 2]]

REF: dict[str, tuple] = {}     # tid -> reference program, for held-out validation


def task_from_ops(tid, base_seq, term=None, inputs=INPUTS, value=1.0) -> Task:
    ref = (tuple(base_seq), term)
    REF[tid] = ref
    return Task(tid, tid, value, [(xs, run_program(ref, xs)) for xs in inputs])


# Training tasks: the 2-gram (feven, psum) recurs across 3 of them; the rest is
# noise. A good learner must pick the *compressing* fragment, not just any frequent.
def training_tasks():
    return [
        task_from_ops("tr1", ("feven", "psum")),
        task_from_ops("tr2", ("feven", "psum"), inputs=[[2, 3, 4], [5, 6, 7, 8]]),
        task_from_ops("tr3", ("feven", "psum"), inputs=[[8, 1, 2], [4, 4, 5]]),
        task_from_ops("tr4", ("mul2",)),                     # noise
        task_from_ops("tr5", ("rev",)),                      # noise
    ]


# Novel held-out tasks: depth-3 in base ops => UNREACHABLE at MAXD=2 before learning.
# After learning macro = (feven, psum), they become (macro, x) at depth 2.
def novel_tasks():
    return [
        task_from_ops("nv1", ("feven", "psum", "rev")),
        task_from_ops("nv2", ("feven", "psum", "srtd")),
    ]


def guardrail_task():
    # depth-3, irreducible (reverse of 2x+2; no two base ops express it), unrelated
    # to (feven, psum): learning that macro must NOT solve it, and base depth-2
    # cannot reach it either.
    return task_from_ops("gr1", ("add1", "mul2", "rev"))


# --------------------------------------------------------------------------- #
# MDL library learner
# --------------------------------------------------------------------------- #
def _expand(names):
    out = []
    for x in names:
        out.extend(LEARNED[x] if x in LEARNED else (x,))
    return tuple(out)


def mine_macro(corpus, min_count=2, max_len=3):
    """Pick the base-op fragment that most reduces total description length.

    savings = count*(len-1)  (each use shrinks by len-1 symbols)
    library cost = len        (store the definition once)
    score = savings - library cost ; require > 0 (must actually compress).
    """
    counts: Counter = Counter()
    for names, _ in corpus:
        base = _expand(names)
        for L in range(2, max_len + 1):
            for i in range(len(base) - L + 1):
                counts[base[i:i + L]] += 1
    best, best_score = None, 0
    for frag, cnt in counts.items():
        if cnt < min_count:
            continue
        score = cnt * (len(frag) - 1) - len(frag)
        if score > best_score:
            best, best_score = frag, score
    return best, best_score


def solve_all(engine, tasks, budget=800):
    """Solve each task, but COUNT it only if the found program GENERALIZES to the
    held-out inputs (vs the reference). This blocks spurious fits on few examples --
    the honest bar for 'solved'."""
    progs, costs = {}, {}
    for t in tasks:
        ok, c, p = engine.solve(t, budget, use_memory=True, reconfigure=False)
        costs[t.tid] = c
        if ok and all(run_program(p, h) == run_program(REF[t.tid], h) for h in HELD):
            progs[t.tid] = p
    return progs, costs


def learn(engine, train, rounds=2, budget=800):
    """Solve training tasks, then fold the best compressing fragment into the DSL."""
    macros = []
    for _ in range(rounds):
        progs, _ = solve_all(engine, train, budget)
        frag, score = mine_macro(list(progs.values()))
        if not frag or score <= 0 or any(seq == frag for _, seq in macros):
            break                                        # nothing new to compress
        name = f"m{len(macros)}"
        engine.add_macro(name, frag)
        macros.append((name, frag))
    return macros


def compress_len(names, macros) -> int:
    """Minimal symbol count when re-expressing a base sequence with the library
    (greedy longest-match). This is the honest MDL accounting -- independent of
    search order or memoization."""
    seqs = sorted((seq for _, seq in macros), key=len, reverse=True)
    i, n = 0, 0
    while i < len(names):
        for seq in seqs:
            if tuple(names[i:i + len(seq)]) == tuple(seq):
                i += len(seq); n += 1; break
        else:
            i += 1; n += 1
    return n


# --------------------------------------------------------------------------- #
# Paradigm report (before vs after) — the three required proofs
# --------------------------------------------------------------------------- #
def paradigm_report(engine=None, verbose=True):
    LEARNED.clear()                                  # isolate the measurement
    train, novel, guard = training_tasks(), novel_tasks(), [guardrail_task()]
    eng = engine or Engine()

    # BEFORE: base vocabulary only
    base_progs, _ = solve_all(eng, train)
    mdl_before = sum(base_length(p) for p in base_progs.values())
    nov_before, _ = solve_all(Engine(), novel)       # fresh, no memory/macros
    grd_before, _ = solve_all(Engine(), guard)

    # LEARN: grow the language
    macros = learn(eng, train)

    # AFTER: re-express the same corpus with the learned library (proper MDL),
    # and test the enlarged vocabulary on novel + guardrail tasks.
    mdl_after_sym = sum(compress_len(p[0], macros) + (1 if p[1] is not None else 0)
                        for p in base_progs.values())
    lib_size = sum(len(seq) for _, seq in macros)
    nov_after, nov_cost = solve_all(eng, novel)
    grd_after, _ = solve_all(eng, guard)

    if verbose:
        print(f"learned macros: {[(n, seq) for n, seq in macros]}\n")
        print(f"1) TRANSFER (novel, depth-3 > base MAXD={MAXD}):")
        print(f"   solved before learning: {len(nov_before)}/{len(novel)}")
        print(f"   solved after  learning: {len(nov_after)}/{len(novel)}   "
              f"<- new capability, not a cache hit")
        print(f"2) MDL COMPRESSION (training corpus description length):")
        print(f"   before: {mdl_before} base symbols")
        print(f"   after : {mdl_after_sym} symbols + {lib_size} library = "
              f"{mdl_after_sym + lib_size}")
        print(f"3) EFFECTIVE DEPTH (a novel solution):")
        for tid, p in sorted(nov_after.items()):
            print(f"   {tid}: program {p[0]} (depth {symbol_length(p)}) "
                  f"== base depth {base_length(p)}")
        print(f"GUARDRAIL (unrelated deep task): solved before {len(grd_before)}, "
              f"after {len(grd_after)} -> learning did NOT magically solve it")
    return {
        "macros": macros,
        "novel_before": len(nov_before), "novel_after": len(nov_after),
        "novel_total": len(novel),
        "mdl_before": mdl_before, "mdl_after": mdl_after_sym + lib_size,
        "guard_before": len(grd_before), "guard_after": len(grd_after),
        "novel_after_progs": nov_after,
    }


def _demo():
    print("== Self-extending abstraction: the engine grows its own language ==")
    print("Caching solutions vs CHANGING the representation to handle the unseen.\n")
    paradigm_report()
    print("\nReading: the macro is learned by COMPRESSION (MDL), it makes tasks beyond")
    print("the base search depth solvable (transfer + effective depth), and it does")
    print("nothing for unrelated work. That is representation change, not memoization.")


def _selftest() -> int:
    r = paradigm_report(verbose=False)
    assert r["macros"], "no macro learned"
    assert r["macros"][0][1] == ("feven", "psum"), r["macros"]      # the compressing frag
    assert r["novel_before"] == 0, r["novel_before"]                # base can't reach depth-3
    assert r["novel_after"] == r["novel_total"], (r["novel_after"], r["novel_total"])
    assert r["mdl_after"] < r["mdl_before"], (r["mdl_after"], r["mdl_before"])
    assert r["guard_before"] == 0 and r["guard_after"] == 0, r       # honest guardrail
    # effective depth: each novel solution is depth<=MAXD but base depth>MAXD
    for p in r["novel_after_progs"].values():
        assert symbol_length(p) <= MAXD < base_length(p), p
    print(f"selftest OK: learned {r['macros'][0][1]}, transfer {r['novel_before']}->"
          f"{r['novel_after']}/{r['novel_total']}, MDL {r['mdl_before']}->{r['mdl_after']}, "
          f"guardrail stays 0")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-extending abstraction (library learning)")
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
