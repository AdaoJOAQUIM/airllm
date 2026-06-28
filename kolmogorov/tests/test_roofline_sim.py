"""Hermetic unit tests for the roofline simulator (plan §9 quality gate)."""
import unittest

from kolmogorov.systems.roofline_sim import (
    HARDWARE, MODELS, analyse, generator_flops_per_weight,
    monte_carlo_crossover,
)


class TestRoofline(unittest.TestCase):

    def test_budget_matches_closed_form(self):
        # K = C_gpu * u / BW_slow, checked against the dataclass property.
        hw = HARDWARE["nvme_gen4"]
        expected = hw.gpu_flops_per_s * hw.utilization / hw.bw_slow_bytes_per_s
        self.assertAlmostEqual(hw.flop_budget_per_byte, expected, places=6)

    def test_rmax_is_half_budget_for_fp16(self):
        # flops/weight = 2*r ; budget/weight = K * 2 bytes ; r_max = budget/2.
        res = analyse("nvme_gen4", "llama2_70b")
        hw = HARDWARE["nvme_gen4"]
        self.assertAlmostEqual(res.max_generator_rank,
                               hw.flop_budget_per_byte * (16 / 8) / 2.0, places=3)

    def test_recompute_decision_is_consistent(self):
        # A rank just under r_max wins; just over loses.
        res = analyse("nvme_gen4", "llama2_70b")
        r = int(res.max_generator_rank)
        budget = HARDWARE["nvme_gen4"].flop_budget_per_byte * (16 / 8)
        self.assertLess(generator_flops_per_weight(r - 1), budget)
        self.assertGreater(generator_flops_per_weight(r + 1), budget)

    def test_slower_storage_gives_bigger_budget(self):
        # The slower the tier, the more FLOPs we may spend per byte.
        sata = analyse("sata_ssd", "llama2_70b").max_generator_rank
        nvme = analyse("nvme_gen4", "llama2_70b").max_generator_rank
        self.assertGreater(sata, nvme)

    def test_reload_time_scales_with_model_size(self):
        small = analyse("nvme_gen4", "llama3_8b").reload_time_s
        big = analyse("nvme_gen4", "llama31_405b").reload_time_s
        self.assertGreater(big, small)

    def test_monte_carlo_percentiles_ordered(self):
        mc = monte_carlo_crossover("nvme_gen4", "llama2_70b", n=5000, seed=1)
        self.assertLess(mc["r_max_p05"], mc["r_max_p50"])
        self.assertLess(mc["r_max_p50"], mc["r_max_p95"])

    def test_determinism(self):
        a = monte_carlo_crossover("nvme_gen4", "llama2_70b", n=2000, seed=7)
        b = monte_carlo_crossover("nvme_gen4", "llama2_70b", n=2000, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
