"""Kernels module - Optimized CUDA/Triton kernels."""

from .attention import FlashAttentionKernel

__all__ = ["FlashAttentionKernel"]
