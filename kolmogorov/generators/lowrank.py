"""G1 baseline generator: truncated-SVD low-rank weight reconstruction.

This is the *weakest* generator family in the plan (§4) and exists precisely as
a conservative lower bound on what C1 can achieve: if even naive low-rank gives
meaningful compression at flat perplexity, the stronger generators (hypernet,
diffusion, hybrid) can only do better. If low-rank already fails, we learn the
structure isn't simple low-rank and pivot generator family — not abandon C1.
"""
from __future__ import annotations

import torch


def low_rank_approx(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the best rank-`rank` approximation of a 2-D `weight` (Eckart-Young)."""
    assert weight.dim() == 2
    w = weight.float()
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    r = max(1, min(rank, S.numel()))
    approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return approx.to(weight.dtype)


def factor_elements(shape: tuple[int, int], rank: int) -> int:
    """Number of stored elements for the rank-r factorization of a d x k matrix."""
    d, k = shape
    r = max(1, min(rank, min(d, k)))
    return r * (d + k)


def effective_bits_per_weight(shape: tuple[int, int], rank: int,
                              store_bitwidth: int = 16) -> float:
    """b_eff for storing this matrix as low-rank factors instead of full."""
    d, k = shape
    return store_bitwidth * factor_elements(shape, rank) / (d * k)
