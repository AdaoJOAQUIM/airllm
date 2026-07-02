"""
Benchmark for the lossless byte-plane codec: measures the effective
bits-per-parameter, the metric to track against the theoretical floors
discussed in docs/ROADMAP_1T.md.

Usage:
    # on a real model shard (any .safetensors file):
    python lossless_bench.py /path/to/model.safetensors

    # or without arguments, on synthetic trained-like weights:
    python lossless_bench.py
"""

import sys
import time

import torch

sys.path.insert(0, '..')

from airllm.lossless import (compress_state_dict_lossless, decompress_state_dict_lossless,
                             compression_report)


def synthetic_state_dict(hidden=2048, ffn=5504, dtype=torch.bfloat16):
    """One transformer-like decoder layer with trained-like weight scales."""
    torch.manual_seed(0)

    def w(shape, std):
        return torch.normal(0, std, shape, dtype=dtype)

    return {
        'self_attn.q_proj.weight': w((hidden, hidden), 0.02),
        'self_attn.k_proj.weight': w((hidden, hidden), 0.02),
        'self_attn.v_proj.weight': w((hidden, hidden), 0.02),
        'self_attn.o_proj.weight': w((hidden, hidden), 0.02),
        'mlp.gate_proj.weight': w((ffn, hidden), 0.02),
        'mlp.up_proj.weight': w((ffn, hidden), 0.02),
        'mlp.down_proj.weight': w((hidden, ffn), 0.02),
        'input_layernorm.weight': torch.ones(hidden, dtype=dtype),
        'post_attention_layernorm.weight': torch.ones(hidden, dtype=dtype),
    }


def main():
    if len(sys.argv) > 1:
        from safetensors.torch import load_file
        print(f"loading {sys.argv[1]} ...")
        state_dict = load_file(sys.argv[1], device='cpu')
    else:
        print("no safetensors file given, using a synthetic transformer layer "
              "(trained-like Gaussian weights)")
        state_dict = synthetic_state_dict()

    report = compression_report(state_dict)

    t0 = time.perf_counter()
    compressed = compress_state_dict_lossless(state_dict)
    t_compress = time.perf_counter() - t0

    t0 = time.perf_counter()
    restored = decompress_state_dict_lossless(compressed)
    t_decompress = time.perf_counter() - t0

    exact = all(torch.equal(state_dict[k], restored[k]) for k in state_dict)

    print()
    print(f"tensors:               {len(state_dict)}")
    print(f"parameters:            {report['n_params']:,}")
    print(f"raw size:              {report['raw_bytes'] / 1e6:.2f} MB "
          f"({report['raw_bits_per_param']:.2f} bits/param)")
    print(f"compressed size:       {report['compressed_bytes'] / 1e6:.2f} MB "
          f"({report['bits_per_param']:.2f} bits/param)")
    print(f"ratio:                 {report['ratio']:.4f} "
          f"({(1 - report['ratio']) * 100:.1f}% saved)")
    print(f"compress throughput:   {report['raw_bytes'] / 1e6 / t_compress:.1f} MB/s")
    print(f"decompress throughput: {report['raw_bytes'] / 1e6 / t_decompress:.1f} MB/s")
    print(f"bit-for-bit exact:     {exact}")

    if not exact:
        raise SystemExit("ERROR: roundtrip was not exact!")


if __name__ == '__main__':
    main()
