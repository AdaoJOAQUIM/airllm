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

    def test_learned_algebra_trains_and_dim4_beats_dim3(self):
        # Short training run: the learned 4D bilinear fold should already be
        # clearly ahead of the 3D one, since it can rediscover the (4D)
        # quaternion product while 3D structurally cannot.
        import numpy as np
        from trinition import benchmark as bm
        from trinition import learn_algebra as la
        rng = np.random.default_rng(0)
        seqs_tr, y_tr = bm.make_task_a(800, 5, rng)
        seqs_te, y_te = bm.make_task_a(500, 5, rng)
        mse3, hist3 = la.train_learned_algebra(
            3, seqs_tr, y_tr, seqs_te, y_te, epochs=200, lr=5e-3, seed=0)
        mse4, hist4 = la.train_learned_algebra(
            4, seqs_tr, y_tr, seqs_te, y_te, epochs=200, lr=5e-3, seed=0)
        self.assertLess(hist4[-1], hist4[0])      # it actually trained
        self.assertLess(mse4, mse3)               # extra dimension helps


class TestLongMemory(unittest.TestCase):
    """The decisive fractional-vs-SSM experiment. Asserts the cross-over: the
    1-parameter fractional order should win the power-law regime and lose the
    exponential regime -- the honest, scoped result."""

    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def test_gl_weights_grad_matches_finite_difference(self):
        import numpy as np
        from trinition import benchmark_longmemory as lm
        order, n = -0.4, 20
        w, dw = lm._gl_weights_grad(order, n)
        eps = 1e-6
        wp, _ = lm._gl_weights_grad(order + eps, n)
        wm, _ = lm._gl_weights_grad(order - eps, n)
        fd = (wp - wm) / (2 * eps)
        self.assertTrue(np.allclose(dw, fd, atol=1e-4))

    def test_fractional_wins_power_loses_exp(self):
        import numpy as np
        from trinition import benchmark_longmemory as lm
        rng = np.random.default_rng(0)
        T, ntr, nte, ep = 48, 1500, 1000, 200
        # power-law regime: fractional (4 params) should beat the equal-budget
        # small SSM (state=2, 7 params).
        Xtr, ytr = lm.make_dataset(ntr, T, "power", rng)
        Xte, yte = lm.make_dataset(nte, T, "power", rng)
        mse_f, pf, q = lm.train_fractional(Xtr, ytr, Xte, yte, memory=T,
                                           epochs=ep, lr=2e-2)
        mse_s2, _ = lm.train_ssm(Xtr, ytr, Xte, yte, state_dim=2,
                                 epochs=ep, lr=2e-2)
        self.assertLess(mse_f, mse_s2)
        # exponential regime: the SSM is exact territory and should beat
        # the fractional power-law kernel.
        Xtr, ytr = lm.make_dataset(ntr, T, "exp", rng)
        Xte, yte = lm.make_dataset(nte, T, "exp", rng)
        mse_f2, _, _ = lm.train_fractional(Xtr, ytr, Xte, yte, memory=T,
                                           epochs=ep, lr=2e-2)
        mse_s2e, _ = lm.train_ssm(Xtr, ytr, Xte, yte, state_dim=2,
                                  epochs=ep, lr=2e-2)
        self.assertLess(mse_s2e, mse_f2)


if __name__ == "__main__":
    unittest.main()
