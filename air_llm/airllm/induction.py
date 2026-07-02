"""
Computable approximations of Solomonoff induction (docs/THEORY.md,
Theorem 7). Pure standard library; no torch required.

Solomonoff's universal inductor -- the Bayesian mixture over all programs
weighted by brevity -- is uncomputable (it consults the halting problem).
Each class here is a published brick that makes a bounded version of it
computable:

- CTWPredictor: Context Tree Weighting (Willems, Shtarkov & Tjalkens
  1995) -- the exact Bayesian mixture over ALL binary tree sources of
  bounded depth, in O(depth) per symbol. The computable miniature of
  Solomonoff's mixture, and the world model inside MC-AIXI-CTW (Veness
  et al. 2011, arXiv:0909.0801).

- mdl_program_search: Levin-style search (1973) over a bounded, total
  expression language -- find the SHORTEST program explaining a sequence,
  then predict with it. Universality is traded for totality, so no
  halting problem arises.

- ncd: Normalized Compression Distance (Cilibrasi & Vitanyi 2005) --
  Kolmogorov similarity approximated with a real compressor.

- predict_by_compression: prediction = compression (Deletang et al. 2023)
  run backwards: the likeliest continuation is the one that compresses
  best against the history.
"""

import itertools
import math
import zlib


# ---------------------------------------------------------------------------
# Context Tree Weighting (Willems, Shtarkov & Tjalkens 1995)
# ---------------------------------------------------------------------------

class _CTWNode:
    __slots__ = ('a', 'b', 'log_pe', 'log_pw', 'children')

    def __init__(self):
        self.a = 0          # zeros seen in this context
        self.b = 0          # ones seen in this context
        self.log_pe = 0.0   # log Krichevsky-Trofimov estimate
        self.log_pw = 0.0   # log weighted (mixture) probability
        self.children = {}


def _logaddexp(x, y):
    if x < y:
        x, y = y, x
    return x + math.log1p(math.exp(y - x))


class CTWPredictor:
    """
    Exact Bayesian mixture over all binary tree sources of depth <= depth.

    update(bit) consumes one bit; predict_one() returns P(next bit = 1);
    logloss_bits() is the total code length so far -- within the
    Willems et al. redundancy bound of the best tree source in hindsight.
    """

    def __init__(self, depth=8):
        self.depth = depth
        self.root = _CTWNode()
        self.history = []

    def _context(self):
        # most recent bit first, zero-padded at the start of the sequence
        ctx = self.history[-self.depth:][::-1]
        return ctx + [0] * (self.depth - len(ctx))

    def _path_nodes(self, create):
        """Nodes from root to leaf along the current context."""
        ctx = self._context()
        path = [self.root]
        node = self.root
        for d in range(self.depth):
            child = node.children.get(ctx[d])
            if child is None:
                if not create:
                    path.append(None)
                    node = _CTWNode()
                    continue
                child = _CTWNode()
                node.children[ctx[d]] = child
            path.append(child)
            node = child
        return ctx, path

    @staticmethod
    def _kt_update(node, bit):
        count = node.b if bit else node.a
        return node.log_pe + math.log((count + 0.5) / (node.a + node.b + 1.0))

    def _new_log_pws(self, path, ctx, bit):
        """Bottom-up recomputation of log_pw along the path (no mutation)."""
        new_pe = []
        for node in path:
            node = node if node is not None else _CTWNode()
            new_pe.append(self._kt_update(node, bit))

        new_pw = [0.0] * len(path)
        for d in range(self.depth, -1, -1):
            node = path[d] if path[d] is not None else _CTWNode()
            if d == self.depth:
                new_pw[d] = new_pe[d]
            else:
                on_bit = ctx[d]
                sibling = node.children.get(1 - on_bit)
                sibling_log_pw = sibling.log_pw if sibling is not None else 0.0
                new_pw[d] = math.log(0.5) + _logaddexp(new_pe[d],
                                                       new_pw[d + 1] + sibling_log_pw)
        return new_pe, new_pw

    def update(self, bit):
        bit = 1 if bit else 0
        ctx, path = self._path_nodes(create=True)
        new_pe, new_pw = self._new_log_pws(path, ctx, bit)
        for d, node in enumerate(path):
            node.log_pe = new_pe[d]
            node.log_pw = new_pw[d]
            if bit:
                node.b += 1
            else:
                node.a += 1
        self.history.append(bit)

    def predict_one(self):
        """P(next bit = 1 | history)."""
        ctx, path = self._path_nodes(create=False)
        _, new_pw = self._new_log_pws(path, ctx, 1)
        return math.exp(new_pw[0] - self.root.log_pw)

    def logloss_bits(self):
        """Total code length of the history under the mixture, in bits."""
        return -self.root.log_pw / math.log(2)

    def update_bits(self, bits):
        for bit in bits:
            self.update(bit)

    def update_bytes(self, data):
        for byte in data:
            for shift in range(7, -1, -1):
                self.update((byte >> shift) & 1)


# ---------------------------------------------------------------------------
# Levin-style MDL program search over a bounded, total DSL
# ---------------------------------------------------------------------------

_OPS = (('+', lambda x, y: x + y),
        ('-', lambda x, y: x - y),
        ('*', lambda x, y: x * y))


def mdl_program_search(sequence, max_size=9, value_cap=10 ** 12):
    """
    Find the smallest expression f(n) (over n, constants 0..3, +, -, *)
    with f(n) == sequence[n] for all n, enumerating by description length
    (node count) -- the MDL/Occam order of Levin search, made total by
    bounding the language.

    Returns {'expr', 'size', 'prediction'} for f(len(sequence)), or None
    if no program within max_size explains the data.
    """
    sequence = list(sequence)
    n_obs = len(sequence)
    if n_obs == 0:
        return None
    probe = list(range(n_obs + 1))  # observed indices plus the query point

    def signature(fn):
        vals = []
        for n in probe:
            v = fn(n)
            if abs(v) > value_cap:
                return None
            vals.append(v)
        return tuple(vals)

    atoms = [('n', lambda n: n)]
    for c in range(4):
        atoms.append((str(c), (lambda c: lambda n: c)(c)))

    by_size = {1: []}
    seen = set()
    for name, fn in atoms:
        sig = signature(fn)
        if sig is not None and sig not in seen:
            seen.add(sig)
            by_size[1].append((name, sig))

    target = tuple(sequence)

    def check(size):
        for name, sig in by_size.get(size, []):
            if sig[:n_obs] == target:
                return {'expr': name, 'size': size, 'prediction': sig[n_obs]}
        return None

    found = check(1)
    if found:
        return found

    for size in range(2, max_size + 1):
        by_size[size] = []
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            for (ln, ls), (rn, rs) in itertools.product(by_size[left_size],
                                                        by_size.get(right_size, [])):
                for op_name, op in _OPS:
                    vals = tuple(op(a, b) for a, b in zip(ls, rs))
                    if any(abs(v) > value_cap for v in vals):
                        continue
                    if vals in seen:
                        continue
                    seen.add(vals)
                    by_size[size].append((f"({ln} {op_name} {rn})", vals))
        found = check(size)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# Kolmogorov complexity via real compressors (Cilibrasi & Vitanyi 2005)
# ---------------------------------------------------------------------------

def _clen(data):
    return len(zlib.compress(data, 9))


def ncd(x, y):
    """Normalized Compression Distance between two byte strings in [0, ~1.1]."""
    cx, cy, cxy = _clen(x), _clen(y), _clen(x + y)
    return (cxy - min(cx, cy)) / max(cx, cy)


def predict_by_compression(history, candidates=None):
    """
    Prediction = compression, run backwards: return the candidate next
    byte whose appended continuation compresses best against the history.
    """
    if candidates is None:
        candidates = sorted(set(history)) or list(range(256))
    scored = [(_clen(history + bytes([c])), c) for c in candidates]
    return min(scored)[1]
