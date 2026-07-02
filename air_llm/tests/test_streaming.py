import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm.streaming import (TensorStreamer, streamed_linear, StreamedLinear,
                              stream_module_from_shard)
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

    def test_streamed_linear_prefetch_identical(self):
        x = torch.normal(0, 1, (5, 64), dtype=torch.float32)
        for block_rows in (7, 128, 300):
            y_seq = streamed_linear(x, self.raw_path, 'proj.weight',
                                    block_rows=block_rows, prefetch=False)
            y_pre = streamed_linear(x, self.raw_path, 'proj.weight',
                                    block_rows=block_rows, prefetch=True)
            self.assertTrue(torch.equal(y_seq, y_pre))

    def test_streamed_linear_module(self):
        layer = StreamedLinear(self.raw_path, 'proj.weight', bias_key='proj.bias',
                               block_rows=64)
        self.assertEqual(layer.in_features, 64)
        self.assertEqual(layer.out_features, 300)

        x = torch.normal(0, 1, (5, 64), dtype=torch.float32)
        expected = x @ self.sd['proj.weight'].t() + self.sd['proj.bias']
        self.assertTrue(torch.equal(layer(x), expected))

    def test_streamed_mlp_matches_dense_model(self):
        # end-to-end: an MLP whose weights live on disk, running inside a
        # normal nn.Module graph, must equal the dense in-RAM model exactly
        torch.manual_seed(1)
        dense = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 16),
        )
        sd = {'fc1.weight': dense[0].weight.data, 'fc1.bias': dense[0].bias.data,
              'fc2.weight': dense[2].weight.data, 'fc2.bias': dense[2].bias.data}
        path = os.path.join(self.tmp.name, 'mlp.safetensors')
        save_file({k: v.contiguous() for k, v in sd.items()}, path)

        streamer = TensorStreamer(path)
        streamed = torch.nn.Sequential(
            StreamedLinear(streamer, 'fc1.weight', bias_key='fc1.bias', block_rows=13),
            torch.nn.ReLU(),
            StreamedLinear(streamer, 'fc2.weight', bias_key='fc2.bias', block_rows=5),
        )

        x = torch.normal(0, 1, (4, 32), dtype=torch.float32)
        with torch.no_grad():
            # nn.Linear uses the fused addmm BLAS kernel whose accumulation
            # order may round differently than blocked matmul; equality is
            # exact w.r.t. the same kernel (see test_streamed_linear_exact),
            # and within float rounding w.r.t. a different one
            self.assertTrue(torch.allclose(streamed(x), dense(x), rtol=1e-6, atol=1e-6))

    def test_streamed_llama_decoder_layer_bit_identical(self):
        # a real transformers Llama decoder layer with all its linears
        # streamed from disk must match the dense layer bit-for-bit
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import (LlamaDecoderLayer,
                                                              LlamaRotaryEmbedding)

        torch.manual_seed(2)
        cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                          num_key_value_heads=2, num_hidden_layers=2, vocab_size=256)
        dense = LlamaDecoderLayer(cfg, layer_idx=0).eval()

        prefix = 'model.layers.0.'
        sd = {prefix + k: v.detach().contiguous() for k, v in dense.state_dict().items()}
        path = os.path.join(self.tmp.name, 'llama_layer.safetensors')
        save_file(sd, path)

        streamed = LlamaDecoderLayer(cfg, layer_idx=0).eval()
        replaced = stream_module_from_shard(streamed, path, prefix=prefix, block_rows=16)
        self.assertEqual(replaced, 7)  # q, k, v, o, gate, up, down

        x = torch.normal(0, 1, (1, 6, 64))
        pos_emb = LlamaRotaryEmbedding(config=cfg)(x, torch.arange(6).unsqueeze(0))
        with torch.no_grad():
            out_dense = dense(x, position_embeddings=pos_emb)
            out_streamed = streamed(x, position_embeddings=pos_emb)
        od = out_dense[0] if isinstance(out_dense, tuple) else out_dense
        ost = out_streamed[0] if isinstance(out_streamed, tuple) else out_streamed
        self.assertTrue(torch.equal(od, ost))

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
