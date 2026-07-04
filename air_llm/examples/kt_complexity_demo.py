"""
E11 - Theorem 10 (docs/KOLMOGOROV.md): time-bounded Kolmogorov complexity
Kt is computable where K is not. mdl_program_search (induction.py) is a
bounded-Kt coder: it enumerates programs by description length within a
budget and HALTS on every input. We show it (a) recovers the shortest
generating program for structured sequences, and (b) terminates with a
finite answer even on an incompressible one -- exactly the behaviour K
cannot have.
"""
import sys
sys.path.insert(0, '..')
from airllm.induction import mdl_program_search

cases = {
    "even numbers   [0,2,4,6,8]": [0, 2, 4, 6, 8],
    "squares        [0,1,4,9,16]": [0, 1, 4, 9, 16],
    "constant       [5,5,5,5]": [5, 5, 5, 5],
    "random-ish     [3,1,4,1,5,9]": [3, 1, 4, 1, 5, 9],
}
for name, seq in cases.items():
    r = mdl_program_search(seq, max_size=9)
    if r:
        print(f"{name:32s} -> Kt-program '{r['expr']}' (size {r['size']}), "
              f"predicts next = {r['prediction']}")
    else:
        print(f"{name:32s} -> no program <= budget (HALTS with 'incompressible "
              f"within budget') -- K would never halt")
print("\nEvery call returned in finite time: Kt is computable (Theorem 10).")
print("K(x) remains uncomputable and untouched -- only the hypothesis")
print("'unbounded time' was deleted.")
