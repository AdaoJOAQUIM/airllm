"""Experimental Trinition algebra + Atangana-style operators sandbox.

This package is *optional* and isolated: importing it pulls in no third-party
dependencies and does not affect AirLLM's inference path. See ``README.md`` in
this directory for what it is and -- importantly -- what it is not.
"""

from .trinition import (
    Trinition,
    StructureConstants,
    make_structure_constants,
    DEFAULT_STRUCTURE,
    zero,
    one,
    from_vector,
)
from .operators import (
    gl_weights,
    fractional_derivative,
    atangana_memory,
)

__all__ = [
    "Trinition",
    "StructureConstants",
    "make_structure_constants",
    "DEFAULT_STRUCTURE",
    "zero",
    "one",
    "from_vector",
    "gl_weights",
    "fractional_derivative",
    "atangana_memory",
]
