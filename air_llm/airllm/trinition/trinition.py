"""
Trinition: a configurable 3D hypercomplex number ``Z = a + b*i + c*j``.

This is an *experimental* sandbox, not part of AirLLM's inference path. It exists
so that the algebraic ideas discussed around "Trinition / Atangana operators" can
be written down concretely and *measured* instead of only described.

A Trinition number is an element of a 3-dimensional algebra over the reals with
basis ``(1, i, j)``. Unlike the complex numbers or quaternions, a closed 3D
algebra has no single canonical multiplication, so the product is defined by an
explicit table of *structure constants* ``C[k, l, m]`` meaning::

    e_k * e_l = sum_m C[k, l, m] * e_m        with  e_0 = 1, e_1 = i, e_2 = j

Different tables give commutative, non-commutative, or non-associative algebras.
The :func:`make_structure_constants` helper builds a parametrized family so that
"controlled deformability" is an actual knob you can turn, and
:meth:`Trinition.commutator` / :meth:`Trinition.associator` let you *measure* how
far a given table is from commuting / associating.

Components ``a, b, c`` are plain Python floats. No third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# A structure-constant table is a 3x3x3 nested list of floats.
StructureConstants = List[List[List[float]]]


def _zeros_3x3x3() -> StructureConstants:
    return [[[0.0, 0.0, 0.0] for _ in range(3)] for _ in range(3)]


def make_structure_constants(
    i_sq: float = -1.0,
    j_sq: float = -1.0,
    ij: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ji: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    alpha: float = 1.0,
) -> StructureConstants:
    """Build a deformable Trinition multiplication table.

    Parameters
    ----------
    i_sq, j_sq:
        Values of ``i*i`` and ``j*j`` as multiples of the unit ``1``.
        ``-1`` mimics the imaginary units of the complex numbers.
    ij, ji:
        The products ``i*j`` and ``j*i`` expressed in the ``(1, i, j)`` basis.
        Making ``ij != ji`` produces a *non-commutative* algebra.
    alpha:
        Deformation parameter in ``[0, 1]``. ``alpha = 0`` collapses the cross
        terms toward a symmetric (more commutative) product; ``alpha = 1`` uses
        ``ij`` / ``ji`` as given. This is the "controlled deformability" dial:
        sweep it and watch :meth:`Trinition.commutator` grow.

    Returns
    -------
    A 3x3x3 table with ``e_0 = 1`` acting as the identity.
    """
    c = _zeros_3x3x3()

    # Identity: 1 * e_l = e_l  and  e_k * 1 = e_k.
    for k in range(3):
        c[0][k][k] = 1.0
        c[k][0][k] = 1.0

    # Squares of the imaginary units.
    c[1][1][0] = i_sq
    c[2][2][0] = j_sq

    # Deform the cross terms. The symmetric average is the "undeformed" product;
    # alpha interpolates from that average out to the asymmetric (ij, ji) pair.
    avg = tuple((p + q) / 2.0 for p, q in zip(ij, ji))
    ij_d = tuple(a + alpha * (p - a) for a, p in zip(avg, ij))
    ji_d = tuple(a + alpha * (q - a) for a, q in zip(avg, ji))
    for m in range(3):
        c[1][2][m] = ij_d[m]
        c[2][1][m] = ji_d[m]

    return c


# A reasonable default: non-commutative, non-associative, complex-like squares.
DEFAULT_STRUCTURE = make_structure_constants()


@dataclass(frozen=True)
class Trinition:
    """A 3D hypercomplex number ``a + b*i + c*j`` over a chosen algebra.

    The multiplication table is carried per-instance (``structure``) so values
    from different algebras are never silently mixed.
    """

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    structure: StructureConstants = field(default_factory=lambda: DEFAULT_STRUCTURE)

    # -- construction helpers ------------------------------------------------
    @property
    def components(self) -> Tuple[float, float, float]:
        return (self.a, self.b, self.c)

    def with_components(self, a: float, b: float, c: float) -> "Trinition":
        return Trinition(a, b, c, self.structure)

    # -- linear structure ----------------------------------------------------
    def __add__(self, other: "Trinition") -> "Trinition":
        self._check(other)
        return Trinition(self.a + other.a, self.b + other.b, self.c + other.c,
                         self.structure)

    def __sub__(self, other: "Trinition") -> "Trinition":
        self._check(other)
        return Trinition(self.a - other.a, self.b - other.b, self.c - other.c,
                         self.structure)

    def __neg__(self) -> "Trinition":
        return Trinition(-self.a, -self.b, -self.c, self.structure)

    def scale(self, s: float) -> "Trinition":
        """Multiply by a real scalar."""
        return Trinition(self.a * s, self.b * s, self.c * s, self.structure)

    def __rmul__(self, s: float) -> "Trinition":
        # Real scalar on the left: ``2.0 * z``.
        if isinstance(s, (int, float)):
            return self.scale(float(s))
        return NotImplemented

    # -- algebra product -----------------------------------------------------
    def __mul__(self, other) -> "Trinition":
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        self._check(other)
        x = self.components
        y = other.components
        cst = self.structure
        out = [0.0, 0.0, 0.0]
        for k in range(3):
            if x[k] == 0.0:
                continue
            for l in range(3):
                coeff = x[k] * y[l]
                if coeff == 0.0:
                    continue
                row = cst[k][l]
                for m in range(3):
                    out[m] += coeff * row[m]
        return Trinition(out[0], out[1], out[2], self.structure)

    # -- non-commutativity / non-associativity probes ------------------------
    def commutator(self, other: "Trinition") -> "Trinition":
        """``[x, y] = x*y - y*x``. Zero iff the two values commute."""
        return self * other - other * self

    def associator(self, y: "Trinition", z: "Trinition") -> "Trinition":
        """``(x*y)*z - x*(y*z)``. Zero iff the triple associates."""
        return (self * y) * z - self * (y * z)

    # -- norms ---------------------------------------------------------------
    def norm(self) -> float:
        """Euclidean norm of the component vector."""
        return (self.a * self.a + self.b * self.b + self.c * self.c) ** 0.5

    # -- misc ----------------------------------------------------------------
    def _check(self, other: "Trinition") -> None:
        if not isinstance(other, Trinition):
            raise TypeError(f"expected Trinition, got {type(other).__name__}")
        if other.structure is not self.structure and other.structure != self.structure:
            raise ValueError("Trinition values come from different algebras")

    def __repr__(self) -> str:
        return f"Trinition({self.a:g} + {self.b:g}i + {self.c:g}j)"


def zero(structure: StructureConstants = DEFAULT_STRUCTURE) -> Trinition:
    return Trinition(0.0, 0.0, 0.0, structure)


def one(structure: StructureConstants = DEFAULT_STRUCTURE) -> Trinition:
    return Trinition(1.0, 0.0, 0.0, structure)


def from_vector(vec: Sequence[float],
                structure: StructureConstants = DEFAULT_STRUCTURE) -> Trinition:
    a, b, c = (list(vec) + [0.0, 0.0, 0.0])[:3]
    return Trinition(a, b, c, structure)
