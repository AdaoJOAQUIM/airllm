"""Hermetic unit tests for the quantizers (no network/GPU)."""
import unittest
import torch

from kolmogorov.generators.quantize import (
    quantize_per_channel, quantize_per_channel_calibrated, is_quantizable,
)


class TestQuantize(unittest.TestCase):

    def test_passthrough_at_16bit(self):
        w = torch.randn(8, 16)
        self.assertTrue(torch.equal(quantize_per_channel(w, 16), w))

    def test_error_decreases_with_bits(self):
        torch.manual_seed(0)
        w = torch.randn(32, 64)
        e8 = (quantize_per_channel(w, 8) - w).pow(2).mean().item()
        e4 = (quantize_per_channel(w, 4) - w).pow(2).mean().item()
        e2 = (quantize_per_channel(w, 2) - w).pow(2).mean().item()
        self.assertLess(e8, e4)
        self.assertLess(e4, e2)

    def test_calibration_no_worse_than_rtn(self):
        # MSE-clip calibration includes clip=1.0 (== RTN), so it can only help.
        torch.manual_seed(1)
        w = torch.randn(32, 64) * torch.tensor([5.0]).repeat(32, 64)  # heavy tails
        e_rtn = (quantize_per_channel(w, 3) - w).pow(2).mean().item()
        e_cal = (quantize_per_channel_calibrated(w, 3) - w).pow(2).mean().item()
        self.assertLessEqual(e_cal, e_rtn + 1e-9)

    def test_calibration_helps_on_heavy_tail(self):
        # Laplace (heavy-tailed) rows with no single dominating outlier: clipping
        # the tail to reduce grid step on the bulk lowers MSE at low bit-width.
        torch.manual_seed(2)
        lap = torch.distributions.Laplace(0.0, 1.0).sample((16, 256))
        e_rtn = (quantize_per_channel(lap, 2) - lap).pow(2).mean().item()
        e_cal = (quantize_per_channel_calibrated(lap, 2) - lap).pow(2).mean().item()
        self.assertLess(e_cal, e_rtn)   # strict gain when no single outlier dominates

    def test_is_quantizable_filters(self):
        self.assertTrue(is_quantizable("layers.0.mlp.dense.weight", torch.randn(128, 128)))
        self.assertFalse(is_quantizable("embed_in.weight", torch.randn(128, 128)))
        self.assertFalse(is_quantizable("ln.weight", torch.randn(128)))
        self.assertFalse(is_quantizable("tiny.weight", torch.randn(8, 8)))


if __name__ == "__main__":
    unittest.main()
