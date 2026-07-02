import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm.quant_cpu import (quantize_tensor_cpu, dequantize_tensor_cpu,
                              compress_state_dict_cpu_quant,
                              decompress_state_dict_cpu_quant, is_cpu_quantized)
from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict


def rmse(a, b):
    return torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item()


class TestCpuQuant(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.w = torch.normal(0, 1, (128, 256), dtype=torch.float16)

    def test_8bit_roundtrip_error(self):
        payload, absmax, meta = quantize_tensor_cpu(self.w, '8bit-cpu')
        restored = dequantize_tensor_cpu(payload, absmax, meta)
        self.assertEqual(restored.dtype, self.w.dtype)
        self.assertEqual(restored.shape, self.w.shape)
        self.assertLess(rmse(self.w, restored), 0.01)

    def test_4bit_roundtrip_error(self):
        payload, absmax, meta = quantize_tensor_cpu(self.w, '4bit-cpu')
        restored = dequantize_tensor_cpu(payload, absmax, meta)
        self.assertEqual(restored.dtype, self.w.dtype)
        self.assertEqual(restored.shape, self.w.shape)
        # same tolerance the repo uses for the bitsandbytes 4bit path
        self.assertLess(rmse(self.w, restored), 0.1)

    def test_4bit_payload_is_half_a_byte_per_weight(self):
        payload, absmax, meta = quantize_tensor_cpu(self.w, '4bit-cpu')
        self.assertEqual(payload.numel(), self.w.numel() // 2)

    def test_error_bound_theorem6(self):
        # elementwise error must respect |w - q(w)| <= absmax * delta, with
        # delta half the largest NF4 codebook gap (docs/THEORY.md, Theorem 6)
        from airllm.quant_cpu import _NF4_CODE
        delta = (_NF4_CODE[1:] - _NF4_CODE[:-1]).max().item() / 2

        payload, absmax, meta = quantize_tensor_cpu(self.w, '4bit-cpu', blocksize=64)
        restored = dequantize_tensor_cpu(payload, absmax, meta)

        err = (self.w.float() - restored.float()).abs().flatten()
        n = self.w.numel()
        padded = torch.zeros(absmax.numel() * 64)
        padded[:n] = err
        per_block_err = padded.reshape(-1, 64).amax(dim=1)
        # small slack for the fp16 cast of the dequantized values
        self.assertTrue(bool((per_block_err <= absmax * delta + 1e-2).all()))

    def test_odd_sizes_and_scalars(self):
        for shape in ((7,), (3, 5), ()):
            t = torch.normal(0, 1, shape, dtype=torch.float32)
            for mode in ('4bit-cpu', '8bit-cpu'):
                payload, absmax, meta = quantize_tensor_cpu(t, mode)
                restored = dequantize_tensor_cpu(payload, absmax, meta)
                self.assertEqual(restored.shape, t.shape)
                self.assertLess(rmse(t, restored), 0.1)

    def test_state_dict_roundtrip_and_pipeline(self):
        sd = {'a': self.w, 'b': torch.normal(0, 1, (64,), dtype=torch.bfloat16)}
        for mode in ('4bit-cpu', '8bit-cpu'):
            compressed = compress_layer_state_dict(dict(sd), mode)
            self.assertTrue(is_cpu_quantized(compressed))
            restored = uncompress_layer_state_dict(compressed)
            self.assertEqual(set(restored.keys()), set(sd.keys()))
            for k in sd:
                self.assertEqual(restored[k].dtype, sd[k].dtype)
                self.assertLess(rmse(sd[k], restored[k]), 0.1)

    def test_8bit_better_than_4bit(self):
        p4, a4, m4 = quantize_tensor_cpu(self.w, '4bit-cpu')
        p8, a8, m8 = quantize_tensor_cpu(self.w, '8bit-cpu')
        self.assertLess(rmse(self.w, dequantize_tensor_cpu(p8, a8, m8)),
                        rmse(self.w, dequantize_tensor_cpu(p4, a4, m4)))


if __name__ == '__main__':
    unittest.main()
