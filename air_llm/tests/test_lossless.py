import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm.lossless import (compress_tensor_lossless, decompress_tensor_lossless,
                             compress_state_dict_lossless, decompress_state_dict_lossless,
                             is_lossless_compressed, compression_report,
                             LOSSLESS_META_SUFFIX)
from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict


class TestLosslessCodec(unittest.TestCase):

    def assert_roundtrip_exact(self, t):
        payload, meta = compress_tensor_lossless(t)
        self.assertEqual(payload.dtype, torch.uint8)
        restored = decompress_tensor_lossless(payload, meta)
        self.assertEqual(restored.dtype, t.dtype)
        self.assertEqual(restored.shape, t.shape)
        # bit-for-bit equality, including NaN payloads and signed zeros
        if t.numel() > 0 and t.dtype.is_floating_point:
            self.assertTrue(torch.equal(t.contiguous().reshape(-1).view(torch.uint8),
                                        restored.contiguous().reshape(-1).view(torch.uint8)))
        else:
            self.assertTrue(torch.equal(t, restored))

    def test_roundtrip_float16(self):
        torch.manual_seed(0)
        self.assert_roundtrip_exact(torch.normal(0, 0.02, (128, 256), dtype=torch.float16))

    def test_roundtrip_bfloat16(self):
        torch.manual_seed(1)
        self.assert_roundtrip_exact(torch.normal(0, 0.02, (128, 256), dtype=torch.bfloat16))

    def test_roundtrip_float32(self):
        torch.manual_seed(2)
        self.assert_roundtrip_exact(torch.normal(0, 0.02, (64, 64), dtype=torch.float32))

    def test_roundtrip_special_values(self):
        t = torch.tensor([0.0, -0.0, float('inf'), float('-inf'), float('nan'), 65504.0],
                         dtype=torch.float16)
        self.assert_roundtrip_exact(t)

    def test_roundtrip_scalar_and_empty(self):
        self.assert_roundtrip_exact(torch.tensor(1.5, dtype=torch.bfloat16))
        self.assert_roundtrip_exact(torch.empty((0, 8), dtype=torch.float16))

    def test_roundtrip_non_contiguous(self):
        torch.manual_seed(3)
        t = torch.normal(0, 0.02, (64, 64), dtype=torch.float16).t()
        self.assertFalse(t.is_contiguous())
        self.assert_roundtrip_exact(t)

    def test_gaussian_weights_actually_compress(self):
        # trained-like weights (small std) have low-entropy exponents:
        # the codec must beat 16 bits/param by a clear margin
        torch.manual_seed(4)
        sd = {'w': torch.normal(0, 0.02, (512, 512), dtype=torch.bfloat16)}
        report = compression_report(sd)
        self.assertLess(report['bits_per_param'], 14.0)
        self.assertEqual(report['raw_bits_per_param'], 16.0)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(5)
        sd = {'a': torch.normal(0, 0.02, (32, 128), dtype=torch.float16),
              'b': torch.normal(0, 0.02, (128,), dtype=torch.bfloat16)}
        compressed = compress_state_dict_lossless(sd)
        self.assertTrue(is_lossless_compressed(compressed))
        self.assertIn('a' + LOSSLESS_META_SUFFIX, compressed)
        restored = decompress_state_dict_lossless(compressed)
        self.assertEqual(set(restored.keys()), set(sd.keys()))
        for k in sd:
            self.assertTrue(torch.equal(sd[k], restored[k]))

    def test_integration_with_layer_compression_pipeline(self):
        # the airllm split/load pipeline entry points must round-trip exactly
        torch.manual_seed(6)
        sd = {'a0': torch.normal(0, 0.02, (32, 128), dtype=torch.float16),
              'a1': torch.normal(0, 0.02, (32, 128), dtype=torch.bfloat16)}

        compressed = compress_layer_state_dict(sd, 'lossless')
        self.assertTrue(all(v.dtype == torch.uint8 for v in compressed.values()))

        restored = uncompress_layer_state_dict(compressed)
        for k in sd:
            self.assertTrue(torch.equal(sd[k], restored[k]))

    def test_safetensors_persist_roundtrip(self):
        # same storage path as the real split/load pipeline
        import tempfile
        from safetensors.torch import save_file, load_file

        torch.manual_seed(7)
        sd = {'w': torch.normal(0, 0.02, (64, 64), dtype=torch.bfloat16)}
        compressed = compress_layer_state_dict(sd, 'lossless')

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'layer.safetensors')
            save_file(compressed, path)
            loaded = load_file(path, device='cpu')
            restored = uncompress_layer_state_dict(loaded)

        self.assertTrue(torch.equal(sd['w'], restored['w']))

    def test_no_compression_passthrough(self):
        sd = {'a0': torch.normal(0, 1, (4, 4), dtype=torch.float16)}
        self.assertIs(compress_layer_state_dict(sd, None), sd)
        self.assertIs(uncompress_layer_state_dict(sd), sd)


if __name__ == '__main__':
    unittest.main()
