"""Tests for the experimental Trinition sandbox.

Pure Python, no third-party dependencies -- runnable with plain ``unittest``.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airllm"))

from trinition import (  # noqa: E402
    Trinition,
    make_structure_constants,
    from_vector,
    one,
    zero,
    gl_weights,
    fractional_derivative,
    atangana_memory,
)


class TestAlgebra(unittest.TestCase):
    def test_identity(self):
        i = Trinition(0, 1, 0)
        self.assertEqual((one() * i).components, i.components)
        self.assertEqual((i * one()).components, i.components)

    def test_i_squared_is_minus_one_by_default(self):
        i = Trinition(0, 1, 0)
        self.assertEqual((i * i).components, (-1.0, 0.0, 0.0))

    def test_distributive_over_addition(self):
        x = from_vector([1, 2, 3])
        y = from_vector([0.5, -1, 2])
        z = from_vector([2, 0, -1])
        left = x * (y + z)
        right = x * y + x * z
        for a, b in zip(left.components, right.components):
            self.assertAlmostEqual(a, b)

    def test_scalar_commutes(self):
        x = from_vector([1, 2, 3])
        self.assertEqual((3.0 * x).components, x.scale(3.0).components)

    def test_default_table_is_non_commutative(self):
        i = Trinition(0, 1, 0)
        j = Trinition(0, 0, 1)
        comm = i.commutator(j)
        self.assertGreater(comm.norm(), 0.0)

    def test_deformation_controls_commutator(self):
        # alpha = 0 -> symmetric cross terms -> i and j commute.
        sym = make_structure_constants(alpha=0.0)
        i = Trinition(0, 1, 0, sym)
        j = Trinition(0, 0, 1, sym)
        self.assertAlmostEqual(i.commutator(j).norm(), 0.0)

        # Sweeping alpha up monotonically increases non-commutativity.
        norms = []
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            st = make_structure_constants(alpha=a)
            ii = Trinition(0, 1, 0, st)
            jj = Trinition(0, 0, 1, st)
            norms.append(ii.commutator(jj).norm())
        self.assertTrue(all(x <= y + 1e-12 for x, y in zip(norms, norms[1:])))
        self.assertGreater(norms[-1], norms[0])

    def test_mixing_algebras_raises(self):
        a = Trinition(1, 0, 0, make_structure_constants(alpha=0.0))
        b = Trinition(1, 0, 0, make_structure_constants(alpha=1.0))
        with self.assertRaises(ValueError):
            _ = a * b


class TestFractionalDerivative(unittest.TestCase):
    def test_weights_recurrence_alpha_one(self):
        # binom(1, k): 1, 1, 0, 0, ...  -> GL weights 1, -1, 0, 0
        w = gl_weights(1.0, 3)
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], -1.0)
        self.assertAlmostEqual(w[2], 0.0)
        self.assertAlmostEqual(w[3], 0.0)

    def test_alpha_one_is_backward_difference(self):
        seq = [from_vector([float(t), 0, 0]) for t in range(5)]
        d = fractional_derivative(seq, alpha=1.0, step=1.0)
        # derivative of a linear ramp is constant 1 (after the first sample).
        for n in range(1, 5):
            self.assertAlmostEqual(d[n].a, 1.0)

    def test_alpha_zero_is_identity(self):
        seq = [from_vector([1, 2, 3]), from_vector([4, 5, 6])]
        d = fractional_derivative(seq, alpha=0.0)
        for orig, got in zip(seq, d):
            for x, y in zip(orig.components, got.components):
                self.assertAlmostEqual(x, y)

    def test_causal_first_entry(self):
        seq = [from_vector([2, 0, 0]), from_vector([9, 0, 0])]
        d = fractional_derivative(seq, alpha=0.5)
        # first entry uses only the first sample: w_0 * Z_0 = Z_0.
        self.assertAlmostEqual(d[0].a, 2.0)


class TestAtanganaMemory(unittest.TestCase):
    def test_zero_retention_is_passthrough(self):
        seq = [from_vector([1, 0, 0]), from_vector([2, 0, 0])]
        out = atangana_memory(seq, retention=0.0, mix=1.0)
        self.assertAlmostEqual(out[0].a, 1.0)
        self.assertAlmostEqual(out[1].a, 2.0)

    def test_memory_accumulates(self):
        seq = [from_vector([1, 0, 0])] * 4
        out = atangana_memory(seq, retention=0.5, mix=1.0)
        # S_n = 1 + 0.5*S_{n-1}: 1, 1.5, 1.75, 1.875
        expected = [1.0, 1.5, 1.75, 1.875]
        for got, exp in zip(out, expected):
            self.assertAlmostEqual(got.a, exp)

    def test_reset_clears_history(self):
        seq = [from_vector([1, 0, 0])] * 4
        out = atangana_memory(seq, retention=0.5, mix=1.0, resets=[2])
        # reset before index 2 restarts the accumulation: 1, 1.5, 1, 1.5
        self.assertAlmostEqual(out[2].a, 1.0)
        self.assertAlmostEqual(out[3].a, 1.5)

    def test_invalid_params(self):
        seq = [from_vector([1, 0, 0])]
        with self.assertRaises(ValueError):
            atangana_memory(seq, retention=1.0)
        with self.assertRaises(ValueError):
            atangana_memory(seq, mix=2.0)


class TestBenchmark(unittest.TestCase):
    """Smoke + sanity checks for the falsification harness. These assert the
    *controls* hold (e.g. the matched algebra solves its task), not a particular
    outcome for Trinition."""

    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def test_quaternion_solves_rotation_control(self):
        import numpy as np
        from trinition import benchmark as bm
        rng = np.random.default_rng(0)
        res = bm.run_task_a(rng, sizes=[400], length=5, n_test=500)
        # The matched algebra must essentially solve exact quaternion composition;
        # the real-diagonal baseline must not. This validates the harness.
        self.assertLess(res["quaternion"][-1], 1e-6)
        self.assertGreater(res["real"][-1], 1e-3)

    def test_deformation_sweep_runs(self):
        import numpy as np
        from trinition import benchmark as bm
        rng = np.random.default_rng(0)
        sweep = bm.sweep_trinition_deformation(
            rng, n_train=400, length=5, n_test=500, alphas=[0.0, 0.5, 1.0])
        self.assertEqual(len(sweep), 3)
        for alpha, mse in sweep:
            self.assertGreaterEqual(mse, 0.0)


if __name__ == "__main__":
    unittest.main()
