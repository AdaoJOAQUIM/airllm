"""
Lossless byte-plane compression of model weights (pure CPU, no GPU needed).

Principle (see ZipNN, Hershcovitch et al. 2024, and DFloat11, "70% Size,
100% Accuracy", arXiv:2504.11651): in trained bfloat16/float16 weights the
sign+exponent byte is highly redundant (~2-3 bits of entropy out of 8),
while mantissa bytes are near-random. Splitting the raw bytes into
per-position planes and entropy-coding each plane separately yields ~25-30%
size reduction with bit-for-bit exact reconstruction.

The entropy coder used here is DEFLATE (zlib, standard library), which gets
close to the empirical entropy of the exponent plane. Planes that don't
compress (mantissa) are stored raw.

Storage format: for each tensor key `k` of the original state dict, the
compressed state dict holds
  - `k`                  : uint8 tensor with the concatenated plane blobs
  - `k + ".lossless.meta"`: uint8 tensor with a JSON header (dtype, shape,
                            per-plane sizes and flags)
so it round-trips through safetensors exactly like the existing 4bit/8bit
paths.
"""

import json
import zlib

import numpy as np
import torch


LOSSLESS_META_SUFFIX = ".lossless.meta"

_FORMAT_VERSION = 1

# zlib level 6 is within ~1% of level 9 on exponent planes but much faster
_ZLIB_LEVEL = 6


def _tensor_raw_bytes(t):
    """Return the tensor's raw little-endian bytes as a 1-D numpy uint8 array."""
    t = t.detach().contiguous().flatten()
    if t.numel() == 0:
        return np.empty(0, dtype=np.uint8)
    return t.view(torch.uint8).numpy()


def _dtype_to_str(dtype):
    # 'torch.bfloat16' -> 'bfloat16'
    return str(dtype).split('.')[-1]


def _str_to_dtype(name):
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype in lossless metadata: {name}")
    return dtype


def compress_tensor_lossless(t):
    """
    Compress a tensor without any loss.

    Returns (payload, meta): payload is a 1-D uint8 torch tensor, meta a
    JSON-serializable dict sufficient to reconstruct the tensor exactly.
    """
    itemsize = t.element_size()
    n = t.numel()
    raw = _tensor_raw_bytes(t)

    planes = []
    blobs = []
    if n > 0:
        raw_mat = raw.reshape(n, itemsize)
        for i in range(itemsize):
            plane_bytes = np.ascontiguousarray(raw_mat[:, i]).tobytes()
            compressed = zlib.compress(plane_bytes, _ZLIB_LEVEL)
            if len(compressed) < len(plane_bytes):
                planes.append({'c': 1, 'n': len(compressed)})
                blobs.append(compressed)
            else:
                planes.append({'c': 0, 'n': len(plane_bytes)})
                blobs.append(plane_bytes)

    payload_bytes = b''.join(blobs)
    payload = torch.from_numpy(np.frombuffer(payload_bytes, dtype=np.uint8).copy())

    meta = {
        'v': _FORMAT_VERSION,
        'dtype': _dtype_to_str(t.dtype),
        'shape': list(t.shape),
        'planes': planes,
    }
    return payload, meta


def decompress_tensor_lossless(payload, meta):
    """Exact inverse of compress_tensor_lossless."""
    if meta.get('v') != _FORMAT_VERSION:
        raise ValueError(f"unsupported lossless format version: {meta.get('v')}")

    dtype = _str_to_dtype(meta['dtype'])
    shape = meta['shape']
    planes = meta['planes']
    itemsize = torch.empty(0, dtype=dtype).element_size()

    n = 1
    for d in shape:
        n *= d

    if len(planes) == 0:
        return torch.empty(shape, dtype=dtype)

    payload_bytes = payload.numpy().tobytes()

    raw_mat = np.empty((n, itemsize), dtype=np.uint8)
    offset = 0
    for i, plane in enumerate(planes):
        blob = payload_bytes[offset: offset + plane['n']]
        offset += plane['n']
        if plane['c']:
            plane_bytes = zlib.decompress(blob)
        else:
            plane_bytes = blob
        raw_mat[:, i] = np.frombuffer(plane_bytes, dtype=np.uint8)

    flat = torch.from_numpy(raw_mat.reshape(-1).copy()).view(dtype)
    return flat.reshape(shape)


def _meta_to_tensor(meta):
    return torch.from_numpy(
        np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8).copy())


def _meta_from_tensor(meta_tensor):
    return json.loads(meta_tensor.numpy().tobytes().decode('utf-8'))


def is_lossless_compressed(state_dict):
    return any(k.endswith(LOSSLESS_META_SUFFIX) for k in state_dict.keys())


def compress_state_dict_lossless(state_dict):
    compressed = {}
    for k, v in state_dict.items():
        payload, meta = compress_tensor_lossless(v)
        compressed[k] = payload
        compressed[k + LOSSLESS_META_SUFFIX] = _meta_to_tensor(meta)
    return compressed


def decompress_state_dict_lossless(state_dict):
    decompressed = {}
    for k, v in state_dict.items():
        if k.endswith(LOSSLESS_META_SUFFIX):
            continue
        meta_key = k + LOSSLESS_META_SUFFIX
        if meta_key not in state_dict:
            raise KeyError(f"lossless metadata missing for tensor: {k}")
        meta = _meta_from_tensor(state_dict[meta_key])
        decompressed[k] = decompress_tensor_lossless(v, meta)
    return decompressed


def compression_report(state_dict):
    """
    Measure how well the lossless codec does on a (raw) state dict.

    Returns a dict with the original size, compressed size, ratio, and the
    effective bits per parameter -- the metric to track against the
    theoretical floors discussed in docs/ROADMAP_1T.md.
    """
    raw_bytes = 0
    n_params = 0
    for v in state_dict.values():
        raw_bytes += v.numel() * v.element_size()
        n_params += v.numel()

    compressed = compress_state_dict_lossless(state_dict)
    compressed_bytes = sum(v.numel() * v.element_size() for v in compressed.values())

    return {
        'n_params': n_params,
        'raw_bytes': raw_bytes,
        'compressed_bytes': compressed_bytes,
        'ratio': compressed_bytes / raw_bytes if raw_bytes else 1.0,
        'raw_bits_per_param': 8.0 * raw_bytes / n_params if n_params else 0.0,
        'bits_per_param': 8.0 * compressed_bytes / n_params if n_params else 0.0,
    }
