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


def quantize_per_channel_calibrated(weight: torch.Tensor, bits: int,
                                    clips=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
                                    ) -> torch.Tensor:
    """Per-row uniform quantization with MSE-optimal clipping.

    For each output channel, search a clip ratio in `clips` (fraction of absmax)
    and keep the one minimizing reconstruction MSE for that row. This is a cheap,
    *weight-only* calibration (no data/Hessian): weaker than GPTQ/AWQ, but a real
    improvement over round-to-nearest, especially at low bit-widths. Labelled
    honestly as MSE-clip calibration.
    """
    assert weight.dim() == 2
    if bits >= 16:
        return weight.clone()
    w = weight.float()
    qmax = 2 ** (bits - 1) - 1
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)

    best = None
    best_err = None
    for c in clips:
        scale = (absmax * c) / qmax
        q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
        deq = q * scale
        err = ((deq - w) ** 2).mean(dim=1, keepdim=True)   # per-row MSE
        if best is None:
            best, best_err = deq, err
        else:
            take = err < best_err
            best = torch.where(take, deq, best)
            best_err = torch.where(take, err, best_err)
    return best.to(weight.dtype)


def is_quantizable(name: str, p: torch.Tensor) -> bool:
    """Large 2-D linear weights only; skip embeddings / final head / norms."""
    if p.dim() != 2 or min(p.shape) < 64:
        return False
    low = name.lower()
    if "embed" in low or "wte" in low or "embed_out" in low or "lm_head" in low:
        return False
    return name.endswith(".weight")
