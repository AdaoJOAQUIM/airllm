"""Deterministic perplexity evaluation (plan §6 / §9).

Fixed, non-overlapping chunking over a held-out text slice so every run is
comparable. CPU-friendly: caps the number of tokens evaluated.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def compute_perplexity(model, input_ids: torch.Tensor, seq_len: int = 512,
                       max_chunks: int | None = None) -> float:
    """Mean token-level perplexity over non-overlapping chunks of `input_ids`.

    input_ids: 1-D LongTensor of a tokenized corpus.
    """
    model.eval()
    n = input_ids.numel()
    n_chunks = n // seq_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)

    total_nll = 0.0
    total_tokens = 0
    for c in range(n_chunks):
        chunk = input_ids[c * seq_len:(c + 1) * seq_len].unsqueeze(0)
        out = model(chunk, labels=chunk)
        # HF returns mean NLL over (seq_len-1) shifted tokens.
        n_tok = seq_len - 1
        total_nll += out.loss.item() * n_tok
        total_tokens += n_tok

    return math.exp(total_nll / total_tokens)
