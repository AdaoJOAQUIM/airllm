"""
Tensor-granularity streaming from safetensors shards (Theorem 3 in
docs/THEORY.md): execute weights larger than RAM by loading one tensor --
or one row-block of one tensor -- at a time.

Built on safetensors' lazy `safe_open` API (zero-copy, mmap-backed reads
of individual tensors and slices), so nothing is loaded until requested
and nothing stays resident after use.

Two primitives:

- TensorStreamer: iterate/fetch the tensors of a shard file one at a time,
  transparently decoding shards saved with compression='lossless'.
- streamed_linear: y = x @ W^T computed exactly in row-blocks of W, with
  peak weight memory `block_rows * in_features * itemsize` instead of the
  full matrix (proof of exactness in docs/THEORY.md, Theorem 3).
"""

from pathlib import Path

import torch
from safetensors import safe_open

from .lossless import (LOSSLESS_META_SUFFIX, _meta_from_tensor,
                       decompress_tensor_lossless)


class TensorStreamer:
    """
    Lazily read tensors from a .safetensors file one at a time.

    Works on raw shards and on shards produced with compression='lossless'
    (the per-tensor payload+meta pair is decoded on the fly; block slicing
    inside a lossless tensor is not supported because the entropy-coded
    stream is not random-access -- use raw shards for streamed_linear).
    """

    def __init__(self, path, device='cpu'):
        self.path = str(Path(path))
        self.device = device
        self._handle = safe_open(self.path, framework='pt', device=device)
        all_keys = set(self._handle.keys())
        self._lossless_keys = {k for k in all_keys if k.endswith(LOSSLESS_META_SUFFIX)}
        self._keys = sorted(k for k in all_keys if k not in self._lossless_keys)

    def keys(self):
        return list(self._keys)

    def is_lossless(self, key):
        return (key + LOSSLESS_META_SUFFIX) in self._lossless_keys

    def get(self, key):
        """Load one tensor (decoding it if lossless-compressed)."""
        tensor = self._handle.get_tensor(key)
        if self.is_lossless(key):
            meta = _meta_from_tensor(self._handle.get_tensor(key + LOSSLESS_META_SUFFIX))
            tensor = decompress_tensor_lossless(tensor, meta)
        return tensor

    def get_slice(self, key):
        """Raw safetensors slice object for block access (raw shards only)."""
        if self.is_lossless(key):
            raise ValueError(f"{key} is lossless-compressed; block slicing needs a raw shard")
        return self._handle.get_slice(key)

    def shape(self, key):
        if self.is_lossless(key):
            meta = _meta_from_tensor(self._handle.get_tensor(key + LOSSLESS_META_SUFFIX))
            return list(meta['shape'])
        return list(self._handle.get_slice(key).get_shape())

    def __iter__(self):
        for k in self._keys:
            yield k, self.get(k)

    def load_into_module(self, module, strict=True):
        """
        Fill a module's parameters/buffers one tensor at a time (peak extra
        memory: one tensor), instead of materializing a full state dict.
        """
        expected = dict(module.named_parameters())
        expected.update(dict(module.named_buffers()))
        loaded = set()
        for key in self._keys:
            if key in expected:
                with torch.no_grad():
                    expected[key].copy_(self.get(key))
                loaded.add(key)
            elif strict:
                raise KeyError(f"unexpected tensor in shard: {key}")
        if strict:
            missing = set(expected.keys()) - loaded
            if missing:
                raise KeyError(f"missing tensors in shard: {sorted(missing)}")
        return loaded


def streamed_linear(x, streamer, weight_key, bias=None, block_rows=1024):
    """
    Exact y = x @ W^T + bias, streaming W from disk in row blocks.

    Peak weight memory is one block (block_rows x in_features) regardless
    of W's size, so out_features can exceed available RAM. Exactness:
    output columns j in [i*k, (i+1)*k) depend only on rows i*k..(i+1)*k of
    W; each is computed once, no arithmetic is reordered.
    """
    if isinstance(streamer, (str, Path)):
        streamer = TensorStreamer(streamer)

    w_slice = streamer.get_slice(weight_key)
    out_features, in_features = w_slice.get_shape()
    if x.shape[-1] != in_features:
        raise ValueError(f"x last dim {x.shape[-1]} != in_features {in_features}")

    out_shape = list(x.shape[:-1]) + [out_features]
    y = torch.empty(out_shape, dtype=x.dtype)

    for start in range(0, out_features, block_rows):
        end = min(start + block_rows, out_features)
        w_block = w_slice[start:end]
        y[..., start:end] = x @ w_block.to(x.dtype).t()
        del w_block

    if bias is not None:
        y += bias.to(y.dtype)
    return y
