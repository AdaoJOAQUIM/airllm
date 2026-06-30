"""Runnable demo for the Trinition sandbox.

Run with::

    python -m airllm.trinition.demo

It prints, for a sweep of the deformation parameter, how non-commutative and
non-associative the algebra becomes, then shows the fractional derivative and
the resettable Atangana memory operator on a short Trinition sequence.
"""

from __future__ import annotations

from .trinition import Trinition, make_structure_constants, from_vector
from .operators import fractional_derivative, atangana_memory


def _sweep_deformation() -> None:
    print("Controlled deformability: commutator/associator norms vs. alpha")
    print(f"  {'alpha':>6} | {'||[i,j]||':>10} | {'||assoc(i,i,j)||':>16}")
    print("  " + "-" * 40)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        st = make_structure_constants(alpha=a)
        i = Trinition(0, 1, 0, st)
        j = Trinition(0, 0, 1, st)
        comm = i.commutator(j).norm()
        assoc = i.associator(i, j).norm()
        print(f"  {a:>6.2f} | {comm:>10.4f} | {assoc:>16.4f}")
    print()


def _fractional() -> None:
    print("Fractional derivative of a ramp Z(t) = (t, 0, 0):")
    seq = [from_vector([float(t), 0, 0]) for t in range(6)]
    for alpha in (1.0, 0.5):
        d = fractional_derivative(seq, alpha=alpha, step=1.0)
        vals = ", ".join(f"{z.a:.3f}" for z in d)
        print(f"  alpha={alpha}: [{vals}]")
    print("  (alpha=1 -> constant slope 1; alpha=0.5 -> long-memory profile)\n")


def _memory() -> None:
    print("Atangana resettable memory on a constant stream, reset at index 3:")
    seq = [from_vector([1, 0, 0]) for _ in range(6)]
    out = atangana_memory(seq, retention=0.5, mix=1.0, resets=[3])
    vals = ", ".join(f"{z.a:.3f}" for z in out)
    print(f"  [{vals}]")
    print("  (accumulates, then drops history at the reset -- the anti-KV-cache)\n")


def main() -> None:
    print("=" * 52)
    print(" Trinition sandbox demo (experimental, not inference)")
    print("=" * 52 + "\n")
    _sweep_deformation()
    _fractional()
    _memory()


if __name__ == "__main__":
    main()
