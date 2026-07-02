"""
Pure-CPU blockwise weight quantization (no bitsandbytes, no CUDA).

Implements the storage side of the existing '4bit'/'8bit' compression
modes for machines without CUDA (Raspberry Pi, plain CPU servers):

- '8bit-cpu': blockwise absmax int8 quantization
  (LLM.int8 storage scheme, Dettmers et al.)
- '4bit-cpu': blockwise absmax NF4 quantization, two weights per byte;
  the 16 codebook values sit at standard-normal quantiles, which is
  information-theoretically optimal for Gaussian-like trained weights
  (QLoRA, arXiv:2305.14314; error bound: docs/THEORY.md, Theorem 6)

Storage format mirrors lossless.py: for each tensor key `k` the
compressed state dict holds `k` (uint8 payload), `k + ".cpuquant.absmax"`
(float32 per-block scales) and `k + ".cpuquant.meta"` (uint8 JSON header),
so shards round-trip through safetensors like the other modes.
"""

import json

import numpy as np
import torch


CPUQUANT_META_SUFFIX = ".cpuquant.meta"
CPUQUANT_ABSMAX_SUFFIX = ".cpuquant.absmax"

_FORMAT_VERSION = 1

_DEFAULT_BLOCKSIZE = {'4bit-cpu': 64, '8bit-cpu': 2048}

# NF4 codebook from QLoRA (Dettmers et al. 2023), ascending
_NF4_CODE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=torch.float32)


def _dtype_to_str(dtype):
    return str(dtype).split('.')[-1]


def _str_to_dtype(name):
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype in cpuquant metadata: {name}")
    return dtype


def _blockwise_normalize(t, blocksize):
    """Flatten, pad to a blocksize multiple, return (blocks, absmax, n)."""
    flat = t.detach().contiguous().flatten().to(torch.float32)
    n = flat.numel()
    n_blocks = max(1, (n + blocksize - 1) // blocksize)
    padded = torch.zeros(n_blocks * blocksize, dtype=torch.float32)
    padded[:n] = flat
    blocks = padded.reshape(n_blocks, blocksize)
    absmax = blocks.abs().amax(dim=1).clamp(min=1e-12)
    return blocks / absmax[:, None], absmax, n


def quantize_tensor_cpu(t, mode, blocksize=None):
    """Quantize a tensor. Returns (payload uint8, absmax float32, meta dict)."""
    if blocksize is None:
        blocksize = _DEFAULT_BLOCKSIZE[mode]

    normalized, absmax, n = _blockwise_normalize(t, blocksize)

    if mode == '8bit-cpu':
        q = torch.round(normalized * 127.0).clamp(-127, 127).to(torch.int8)
        payload = q.flatten().view(torch.uint8)
    elif mode == '4bit-cpu':
        # nearest NF4 codebook index per weight, then two indices per byte
        idx = torch.searchsorted(_NF4_CODE, normalized.flatten().contiguous())
        idx = idx.clamp(max=15)
        lower = (idx - 1).clamp(min=0)
        pick_lower = (normalized.flatten() - _NF4_CODE[lower]).abs() < \
                     (_NF4_CODE[idx] - normalized.flatten()).abs()
        idx = torch.where(pick_lower, lower, idx).to(torch.uint8)
        if idx.numel() % 2:
            idx = torch.cat([idx, torch.zeros(1, dtype=torch.uint8)])
        pairs = idx.reshape(-1, 2)
        payload = (pairs[:, 0] << 4) | pairs[:, 1]
    else:
        raise ValueError(f"unknown cpu quantization mode: {mode}")

    meta = {
        'v': _FORMAT_VERSION,
        'mode': mode,
        'dtype': _dtype_to_str(t.dtype),
        'shape': list(t.shape),
        'blocksize': blocksize,
        'n': n,
    }
    return payload.contiguous(), absmax, meta


def dequantize_tensor_cpu(payload, absmax, meta):
    """Inverse of quantize_tensor_cpu (up to the Theorem 6 error bound)."""
    if meta.get('v') != _FORMAT_VERSION:
        raise ValueError(f"unsupported cpuquant format version: {meta.get('v')}")

    mode = meta['mode']
    blocksize = meta['blocksize']
    n = meta['n']
    dtype = _str_to_dtype(meta['dtype'])

    if mode == '8bit-cpu':
        normalized = payload.view(torch.int8).to(torch.float32) / 127.0
    elif mode == '4bit-cpu':
        idx = torch.stack([payload >> 4, payload & 0xF], dim=1).flatten().long()
        normalized = _NF4_CODE[idx]
    else:
        raise ValueError(f"unknown cpu quantization mode: {mode}")

    n_blocks = absmax.numel()
    normalized = normalized[:n_blocks * blocksize].reshape(n_blocks, blocksize)
    flat = (normalized * absmax[:, None].to(torch.float32)).flatten()[:n]
    return flat.to(dtype).reshape(meta['shape'])


def _meta_to_tensor(meta):
    return torch.from_numpy(
        np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8).copy())


def _meta_from_tensor(meta_tensor):
    return json.loads(meta_tensor.numpy().tobytes().decode('utf-8'))


def is_cpu_quantized(state_dict):
    return any(k.endswith(CPUQUANT_META_SUFFIX) for k in state_dict.keys())


def compress_state_dict_cpu_quant(state_dict, mode, blocksize=None):
    compressed = {}
    for k, v in state_dict.items():
        payload, absmax, meta = quantize_tensor_cpu(v, mode, blocksize)
        compressed[k] = payload
        compressed[k + CPUQUANT_ABSMAX_SUFFIX] = absmax
        compressed[k + CPUQUANT_META_SUFFIX] = _meta_to_tensor(meta)
    return compressed


def decompress_state_dict_cpu_quant(state_dict):
    decompressed = {}
    for k, v in state_dict.items():
        if k.endswith(CPUQUANT_META_SUFFIX) or k.endswith(CPUQUANT_ABSMAX_SUFFIX):
            continue
        meta_key = k + CPUQUANT_META_SUFFIX
        if meta_key not in state_dict:
            raise KeyError(f"cpuquant metadata missing for tensor: {k}")
        meta = _meta_from_tensor(state_dict[meta_key])
        decompressed[k] = dequantize_tensor_cpu(v, state_dict[k + CPUQUANT_ABSMAX_SUFFIX], meta)
    return decompressed
