"""Per-output-channel uniform quantization — an operational H_eps probe.

The minimal bits/weight that keeps output distortion (perplexity) below a
threshold is a direct, faithful estimate of the metric-entropy density
H_eps / n_params for the model's function class (PROOFS_KOLMOGOROV.md, Cor 3).
Unlike naive low-rank (refuted in Stage 0), quantization is known to work and so
is the right instrument to *measure* compressibility rather than to fail.

Symmetric, per-row (per output channel) uniform quantization to `bits` bits.
"""
from __future__ import annotations

import torch


def quantize_per_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Round a 2-D weight to a `bits`-bit symmetric per-row grid; return dequant."""
    assert weight.dim() == 2
    if bits >= 16:
        return weight.clone()
    w = weight.float()
    qmax = 2 ** (bits - 1) - 1                      # symmetric signed grid
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return (q * scale).to(weight.dtype)


def is_quantizable(name: str, p: torch.Tensor) -> bool:
    """Large 2-D linear weights only; skip embeddings / final head / norms."""
    if p.dim() != 2 or min(p.shape) < 64:
        return False
    low = name.lower()
    if "embed" in low or "wte" in low or "embed_out" in low or "lm_head" in low:
        return False
    return name.endswith(".weight")
