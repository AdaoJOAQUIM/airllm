"""
E5 - empirical verification of Theorem P' (docs/PROOFS.md):
(a) amortized description: program bits are independent of domain size,
(b) the exponential program-vs-weights ratio,
(c) library growth reduces the search EXPONENT, not just a constant.

Concrete instantiation: integer functions, primitives
inc(x)=x+1, dbl(x)=2x, sqr(x)=x^2. Tasks are chain compositions,
checked on probe points. Search = exhaustive enumeration of chains in
increasing length (the Levin/MDL order of induction.py).
"""

import itertools
from math import ceil, log2


PRIMITIVES = {
    'inc': lambda x: x + 1,
    'dbl': lambda x: 2 * x,
    'sqr': lambda x: x * x,
}

PROBES = list(range(7))


def run_chain(chain, x):
    for f in chain:
        x = f(x)
    return x


def signature(fn):
    return tuple(fn(x) for x in PROBES)


def search(target_sig, library, max_len):
    """Enumerate chains in increasing length; count candidates evaluated."""
    names, fns = list(library.keys()), list(library.values())
    evaluated = 0
    for length in range(1, max_len + 1):
        for combo in itertools.product(range(len(fns)), repeat=length):
            evaluated += 1
            chain = [fns[i] for i in combo]
            if tuple(run_chain(chain, x) for x in PROBES) == target_sig:
                return [names[i] for i in combo], evaluated
    return None, evaluated


def main():
    # ---- (c) search-exponent reduction ----
    # task 1: h(x) = sqr(dbl(inc(x))) -- a length-3 chain, found from base
    h = lambda x: (2 * (x + 1)) ** 2
    base = dict(PRIMITIVES)
    found1, cost1 = search(signature(h), base, max_len=3)
    print(f"task 1, base library (l={len(base)}): found "
          f"{' o '.join(reversed(found1))} after {cost1} candidates")

    # task 2: h(h(x)) -- a length-6 chain over the base primitives
    target2 = lambda x: h(h(x))
    tsig2 = signature(target2)

    _, cost_base = search(tsig2, base, max_len=6)
    print(f"task 2, base library:  length-6 chain found after "
          f"{cost_base} candidates (bound 2*l^k = {2 * len(base) ** 6})")

    # the DreamCoder move: compress task 1's solution into the library
    grown = dict(PRIMITIVES, h=h)
    found2, cost_grown = search(tsig2, grown, max_len=2)
    print(f"task 2, grown library (l'={len(grown)}): found "
          f"{' o '.join(reversed(found2))} after {cost_grown} candidates "
          f"(bound 2*l'^2 = {2 * len(grown) ** 2})")
    print(f"search reduction: {cost_base}/{cost_grown} = "
          f"{cost_base / cost_grown:.1f}x  (exponent k=6 -> 2, Theorem P'c)")

    # ---- (a)+(b) description-length accounting, exact arithmetic ----
    n_bits_domain = 20                  # domain X = 2^20 inputs
    b = 32                              # 32-bit outputs
    ell, k = len(base), 3
    program_bits = k * ceil(log2(ell + 1))
    table_bits = (2 ** n_bits_domain) * b
    kappa = 2.0                         # Theorem K regime
    params_needed = table_bits / kappa
    print(f"\ndomain |X| = 2^{n_bits_domain}, outputs {b}-bit:")
    print(f"  program encoding  (P'a): {program_bits} bits per task")
    print(f"  fact-table storage (P'b): {table_bits:,.0f} bits "
          f"~= {params_needed:,.0f} params at kappa=2")
    print(f"  ratio: {table_bits / program_bits:,.0f}x  -- exponential in n, "
          f"as proven.")


if __name__ == '__main__':
    main()
