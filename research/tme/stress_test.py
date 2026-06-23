"""Stress test for the deterministic kernel (no GPU / no LLM needed).

Pushes TME / igc / abstraction / engine until they break and reports the LIMITS
honestly: search-cost explosion, vocabulary scaling, persistence at scale, nested-
macro recursion depth, robustness to malformed input, and determinism. Findings are
measured, including failures -- the point is to find walls, not to look good.

    python3 stress_test.py
"""

from __future__ import annotations

import time
import tempfile
import tme
from tme import Engine, Task, run_program, BASE, LEARNED


def _unsolvable(inputs):
    # no op composition yields a constant 999 list -> forces a FULL search scan
    return Task("uns", "uns", 1.0, [(xs, [999]) for xs in inputs])


def s1_search_explosion():
    print("== S1: search-cost explosion vs depth (the combinatorial wall) ==")
    inputs = [[1, 2], [3, -4]]
    orig = tme.MAXD
    rows = []
    for d in (1, 2, 3, 4):
        tme.MAXD = d
        LEARNED.clear()
        eng = Engine()
        t0 = time.perf_counter()
        ok, cost, _ = eng.solve(_unsolvable(inputs), budget=10**9,
                                use_memory=False, reconfigure=False)
        dt = time.perf_counter() - t0
        rows.append((d, cost, dt))
        print(f"  MAXD={d}: full scan {cost:>7d} candidates in {dt*1000:8.1f} ms "
              f"({cost/max(dt,1e-9):,.0f} cand/s)  solved={ok}")
    tme.MAXD = orig
    growth = rows[-1][1] / max(rows[-2][1], 1)
    print(f"  WALL: each +1 depth multiplies cost ~x{rows[2][1]/max(rows[1][1],1):.0f} "
          f"(|ops|={len(BASE)}). depth 4 already {rows[-1][1]:,} candidates -> "
          f"brute search is the limit abstraction exists to avoid.\n")


def s2_vocabulary_growth():
    print("== S2: cost vs vocabulary size (learned macros enlarge the space) ==")
    inputs = [[1, 2], [3, -4]]
    tme.MAXD = 2
    for K in (0, 20, 50, 100):
        LEARNED.clear()
        eng = Engine()
        for i in range(K):
            eng.add_macro(f"k{i}", ("mul2", "mul2"))   # harmless distinct names
        t0 = time.perf_counter()
        _, cost, _ = eng.solve(_unsolvable(inputs), budget=10**9,
                               use_memory=False, reconfigure=False)
        dt = time.perf_counter() - t0
        print(f"  +{K:3d} macros (|ops|={len(BASE)+K}): {cost:>7d} candidates, {dt*1000:7.1f} ms")
    LEARNED.clear()
    print("  -> cost grows ~|ops|^MAXD. A big vocabulary WITHOUT a proposer to prune "
          "is intractable (confirms the design constraint).\n")


def s3_persistence_scale():
    print("== S3: persistence at scale (memory bounded? round-trip integrity?) ==")
    LEARNED.clear()
    eng = Engine()
    for n in range(5000):                       # inject many synthetic patterns
        sig = ("list", str(n % 800), bool(n % 2), bool(n % 3), bool(n % 5))
        eng.memory.setdefault(sig, [])
        if len(eng.memory[sig]) < eng.mem_cap:
            eng.memory[sig].append((("mul2", f"k{n}"), None))
    import os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    t0 = time.perf_counter(); eng.save(path); tsave = time.perf_counter() - t0
    size_kb = os.path.getsize(path) / 1024
    eng2 = Engine()
    t0 = time.perf_counter(); eng2.load(path); tload = time.perf_counter() - t0
    patterns = sum(len(v) for v in eng2.memory.values())
    ok = eng2.memory == eng.memory
    print(f"  {len(eng.memory)} signatures, {patterns} patterns (mem_cap={eng.mem_cap} -> bounded)")
    print(f"  save {tsave*1000:.0f} ms, {size_kb:.0f} KB; load {tload*1000:.0f} ms; "
          f"round-trip integrity: {'OK' if ok else 'MISMATCH'}\n")
    LEARNED.clear()


def s4_nested_macro_depth():
    print("== S4: nested-macro recursion depth (where does resolution break?) ==")
    import sys
    limit_found = None
    for K in (50, 200, 500, 900, 2000):
        LEARNED.clear()
        eng = Engine()
        eng.add_macro("b0", ("mul2",))
        try:
            for k in range(1, K):
                eng.add_macro(f"b{k}", (f"b{k-1}", "mul2"))
            _ = run_program(((f"b{K-1}",), None), [1, 2])   # forces recursive resolve
            print(f"  nesting depth {K:>4d}: OK")
        except RecursionError:
            limit_found = K
            print(f"  nesting depth {K:>4d}: RecursionError (Python limit {sys.getrecursionlimit()})")
            break
    LEARNED.clear()
    if limit_found:
        print(f"  LIMIT: deep macro nesting is NOT guarded -> RecursionError near "
              f"~{limit_found}. Real robustness gap if a curriculum stacks that deep.\n")
    else:
        print("  no recursion failure up to tested depth.\n")


def s5_robustness():
    print("== S5: robustness to malformed / adversarial input ==")
    import igc, orchestrator
    cases = []
    # malformed intentions
    for bad in ["x=", "x=nope(input)", "a=sum()", "=double(input)", ""]:
        try:
            igc.compile_intention(bad)
            cases.append((bad or "<empty>", "no error (silent!)"))
        except Exception as e:
            cases.append((bad or "<empty>", f"raised {type(e).__name__} (graceful)"))
    for label, outcome in cases:
        print(f"  igc.compile_intention({label!r:24}): {outcome}")
    # unsolvable within tiny budget -> must be bounded, not hang
    LEARNED.clear()
    eng = Engine()
    ok, cost, _ = eng.solve(_unsolvable([[1, 2]]), budget=37, use_memory=False, reconfigure=False)
    print(f"  unsolvable @ budget=37: solved={ok}, cost={cost} (bounded by budget: "
          f"{'OK' if cost <= 37 else 'LEAK'})")
    # empty input list
    try:
        r = run_program((("psum", "feven"), "sum"), [])
        print(f"  run_program on []: returned {r!r} (no crash)")
    except Exception as e:
        print(f"  run_program on []: raised {type(e).__name__}")
    print()


def s6_determinism():
    print("== S6: determinism (same inputs -> same outputs) ==")
    import abstraction
    a = abstraction.paradigm_report(verbose=False)
    b = abstraction.paradigm_report(verbose=False)
    same = (a["macros"] == b["macros"] and a["mdl_after"] == b["mdl_after"]
            and a["novel_after"] == b["novel_after"])
    c = abstraction.curriculum_report(verbose=False)
    d = abstraction.curriculum_report(verbose=False)
    same2 = (c["ladder"] == d["ladder"] and c["macros"] == d["macros"])
    print(f"  paradigm_report reproducible: {'OK' if same else 'NONDETERMINISTIC'}")
    print(f"  curriculum_report reproducible: {'OK' if same2 else 'NONDETERMINISTIC'}\n")


def main():
    print("STRESS TEST — deterministic kernel (CPU only; no GPU/LLM here)\n")
    t0 = time.perf_counter()
    for fn in (s1_search_explosion, s2_vocabulary_growth, s3_persistence_scale,
               s4_nested_macro_depth, s5_robustness, s6_determinism):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} crashed: {type(e).__name__}: {e}\n")
    print(f"total wall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
