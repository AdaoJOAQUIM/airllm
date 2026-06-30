"""
Sequence operators over Trinition values.

Two operators from the "Atangana / fractional" discussion, written as concrete,
testable functions over sequences of :class:`~trinition.Trinition` numbers:

* :func:`fractional_derivative` -- a Grünwald-Letnikov discrete fractional
  derivative of order ``alpha``. For ``alpha = 1`` it reduces (up to the step
  size) to a backward finite difference, i.e. ordinary "rate of change"; for
  fractional ``alpha`` it mixes in the whole history with algebraically decaying
  weights. This is the "predictive fractional derivative" made concrete.

* :func:`atangana_memory` -- a resettable fading-memory integral. Unlike a raw
  KV-cache, which simply accumulates, this operator weights the past by a
  retention factor and supports hard *resets* (initial-condition compatibility),
  so old context can be cleared instead of polluting everything downstream.

Honest scope note: these are mathematically well-defined operators you can run
and measure. They are *not* a validated replacement for attention, and nothing
here claims to outperform a transformer. They are a sandbox for experimenting
with the ideas numerically.

Pure Python; no third-party dependencies.
"""

from __future__ import annotations

from typing import List, Sequence

from .trinition import Trinition, zero


def gl_weights(alpha: float, n: int) -> List[float]:
    """Grünwald-Letnikov coefficients ``w_k = (-1)^k * binom(alpha, k)``.

    Computed by the stable recurrence ``w_0 = 1``,
    ``w_k = w_{k-1} * (alpha - k + 1) / k`` for ``k = 1..n``.

    Returns ``n + 1`` weights.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    weights = [1.0]
    w = 1.0
    for k in range(1, n + 1):
        # Recurrence for (-1)^k * binom(alpha, k): folds the alternating sign in.
        w = w * (k - 1.0 - alpha) / k
        weights.append(w)
    return weights


def fractional_derivative(seq: Sequence[Trinition],
                          alpha: float,
                          step: float = 1.0) -> List[Trinition]:
    """Discrete fractional derivative of a Trinition sequence.

    ``D^alpha Z(t_n) ~= step**(-alpha) * sum_{k=0}^{n} w_k * Z(t_{n-k})``

    Parameters
    ----------
    seq:
        Sequence of Trinition values sampled at uniform spacing ``step``.
    alpha:
        Derivative order. ``alpha = 1`` -> backward difference (ordinary
        derivative); ``0 < alpha < 1`` -> long-memory fractional derivative;
        ``alpha = 0`` -> identity.
    step:
        Sample spacing ``h > 0``.

    Returns
    -------
    A list the same length as ``seq``; entry ``n`` is the fractional derivative
    evaluated using samples ``0..n`` (causal: only the past is used).
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    seq = list(seq)
    if not seq:
        return []
    structure = seq[0].structure
    scale = step ** (-alpha)
    out: List[Trinition] = []
    for n in range(len(seq)):
        w = gl_weights(alpha, n)
        acc = zero(structure)
        for k in range(n + 1):
            acc = acc + seq[n - k].scale(w[k])
        out.append(acc.scale(scale))
    return out


def atangana_memory(seq: Sequence[Trinition],
                    retention: float = 0.5,
                    mix: float = 0.5,
                    resets: Sequence[int] = (),) -> List[Trinition]:
    """Resettable fading-memory integral -- an "anti-KV-cache" operator.

    The state blends the current value with a geometrically-decayed running
    memory of the past::

        M_n = (1 - mix) * Z_n + mix * S_n
        S_n = Z_n + retention * S_{n-1}      (S reset to 0 at reset points)

    Parameters
    ----------
    seq:
        Sequence of Trinition values.
    retention:
        Memory decay in ``[0, 1)``. ``0`` keeps only the present; values closer
        to ``1`` keep longer history. Must be < 1 so memory stays bounded.
    mix:
        How much of the output comes from accumulated memory vs. the raw current
        value, in ``[0, 1]``.
    resets:
        Indices at which memory is cleared *before* incorporating that step --
        the "initial condition" reset that a plain cache cannot express. Use
        this to drop stale context instead of letting it bleed forward.

    Returns
    -------
    A list the same length as ``seq`` of memory-filtered Trinition values.
    """
    if not (0.0 <= retention < 1.0):
        raise ValueError("retention must be in [0, 1)")
    if not (0.0 <= mix <= 1.0):
        raise ValueError("mix must be in [0, 1]")
    seq = list(seq)
    if not seq:
        return []
    reset_at = set(int(r) for r in resets)
    structure = seq[0].structure
    state = zero(structure)
    out: List[Trinition] = []
    for n, z in enumerate(seq):
        if n in reset_at:
            state = zero(structure)
        state = z + state.scale(retention)
        out.append(z.scale(1.0 - mix) + state.scale(mix))
    return out
