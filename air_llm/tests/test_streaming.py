import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm.streaming import TensorStreamer, streamed_linear
from airllm.utils import compress_layer_state_dict


class TestStreaming(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.sd = {
            'proj.weight': torch.normal(0, 0.02, (300, 64), dtype=torch.float32),
            'proj.bias': torch.normal(0, 0.02, (300,), dtype=torch.float32),
        }
        self.raw_path = os.path.join(self.tmp.name, 'raw.safetensors')
        save_file(self.sd, self.raw_path)

        self.lossless_path = os.path.join(self.tmp.name, 'lossless.safetensors')
        save_file(compress_layer_state_dict(dict(self.sd), 'lossless'), self.lossless_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_streamer_keys_hide_metadata(self):
        streamer = TensorStreamer(self.lossless_path)
        self.assertEqual(sorted(streamer.keys()), sorted(self.sd.keys()))

    def test_streamer_get_raw_and_lossless_identical(self):
        raw = TensorStreamer(self.raw_path)
        lossless = TensorStreamer(self.lossless_path)
        for k in self.sd:
            self.assertTrue(torch.equal(raw.get(k), self.sd[k]))
            self.assertTrue(torch.equal(lossless.get(k), self.sd[k]))
            self.assertEqual(raw.shape(k), list(self.sd[k].shape))
            self.assertEqual(lossless.shape(k), list(self.sd[k].shape))

    def test_streamed_linear_exact(self):
        x = torch.normal(0, 1, (5, 64), dtype=torch.float32)
        expected = x @ self.sd['proj.weight'].t() + self.sd['proj.bias']

        for block_rows in (7, 128, 300, 1024):  # non-divisible, small, exact, larger
            y = streamed_linear(x, self.raw_path, 'proj.weight',
                                bias=self.sd['proj.bias'], block_rows=block_rows)
            self.assertTrue(torch.equal(y, expected),
                            f"mismatch for block_rows={block_rows}")

    def test_streamed_linear_batched_input(self):
        x = torch.normal(0, 1, (2, 3, 64), dtype=torch.float32)
        expected = x @ self.sd['proj.weight'].t()
        y = streamed_linear(x, self.raw_path, 'proj.weight', block_rows=64)
        self.assertTrue(torch.equal(y, expected))

    def test_streamed_linear_rejects_lossless_shard(self):
        x = torch.normal(0, 1, (5, 64), dtype=torch.float32)
        with self.assertRaises(ValueError):
            streamed_linear(x, self.lossless_path, 'proj.weight')

    def test_streamed_linear_rejects_bad_input_dim(self):
        x = torch.normal(0, 1, (5, 65), dtype=torch.float32)
        with self.assertRaises(ValueError):
            streamed_linear(x, self.raw_path, 'proj.weight')

    def test_load_into_module(self):
        module = torch.nn.Linear(64, 300)
        for path in (self.raw_path, self.lossless_path):
            streamer = TensorStreamer(path)
            # shard keys are 'proj.weight'/'proj.bias'; remap for the module
            module_sd_keys = {'proj.weight': 'weight', 'proj.bias': 'bias'}
            expected = dict(module.named_parameters())
            for shard_key, param_key in module_sd_keys.items():
                with torch.no_grad():
                    expected[param_key].copy_(streamer.get(shard_key))
            self.assertTrue(torch.equal(module.weight.data, self.sd['proj.weight']))
            self.assertTrue(torch.equal(module.bias.data, self.sd['proj.bias']))

    def test_load_into_module_strict(self):
        class Proj(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(64, 300)

        module = Proj()
        streamer = TensorStreamer(self.lossless_path)
        loaded = streamer.load_into_module(module)
        self.assertEqual(loaded, set(self.sd.keys()))
        self.assertTrue(torch.equal(module.proj.weight.data, self.sd['proj.weight']))

        # a module expecting more tensors than the shard has must fail loudly
        class Bigger(Proj):
            def __init__(self):
                super().__init__()
                self.extra = torch.nn.Linear(4, 4)

        with self.assertRaises(KeyError):
            TensorStreamer(self.lossless_path).load_into_module(Bigger())


if __name__ == '__main__':
    unittest.main()
