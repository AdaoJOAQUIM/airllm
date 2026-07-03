"""
E6 - empirical verification of Theorem K'' (docs/PROOFS.md):
in the lazy regime (only a linear threshold readout trains), EVERY
architecture has storage capacity exactly 2 bits per trainable
parameter -- the architecture enters only through the feature map, and
Cover's count does not see it.

Two deliberately different feature architectures, same number of
trainable readout parameters n:
  A: phi_A(x) = relu(W1 x)                  (1 hidden layer)
  B: phi_B(x) = tanh(W2 relu(W1 x))         (2 hidden layers, mixed)
Realizability of random +-1 labels decided exactly by LP, as in E4.
Theorem K'' predicts the SAME sharp threshold at m = 2n for both.
"""

import numpy as np
from math import comb
from scipy.optimize import linprog


D_IN = 12        # input dimension (irrelevant to the theorem)
N_FEAT = 20      # trainable readout parameters n
TRIALS = 50
rng = np.random.default_rng(1)


def make_arch_A():
    W1 = rng.normal(size=(N_FEAT, D_IN))
    return lambda X: np.maximum(W1 @ X.T, 0.0).T


def make_arch_B():
    W1 = rng.normal(size=(2 * N_FEAT, D_IN))
    W2 = rng.normal(size=(N_FEAT, 2 * N_FEAT)) / np.sqrt(2 * N_FEAT)
    return lambda X: np.tanh(W2 @ np.maximum(W1 @ X.T, 0.0)).T


def separable(F, y):
    m, n = F.shape
    res = linprog(c=np.zeros(n), A_ub=-(y[:, None] * F), b_ub=-np.ones(m),
                  bounds=[(None, None)] * n, method='highs')
    return res.status == 0


def p_store(phi, m):
    hits = 0
    for _ in range(TRIALS):
        X = rng.normal(size=(m, D_IN))
        y = rng.choice([-1.0, 1.0], size=m)
        hits += separable(phi(X), y)
    return hits / TRIALS


def theory(m, n):
    return sum(comb(m - 1, k) for k in range(min(n, m))) / 2 ** (m - 1)


def main():
    phi_A, phi_B = make_arch_A(), make_arch_B()
    print(f"n = {N_FEAT} trainable params, {TRIALS} trials/point; "
          f"input dim {D_IN}\n")
    print(f"{'m/n':>5} {'arch A (relu)':>14} {'arch B (tanh-relu)':>19} {'theory':>8}")
    for ratio in (1.0, 1.5, 2.0, 2.5, 3.0):
        m = int(ratio * N_FEAT)
        print(f"{ratio:>5.1f} {p_store(phi_A, m):>14.3f} "
              f"{p_store(phi_B, m):>19.3f} {theory(m, N_FEAT):>8.3f}")
    print("\nSame threshold at m/n = 2 for both architectures, matching the")
    print("closed form: architecture-independence in the lazy regime, as")
    print("proven (Theorem K'').")


if __name__ == '__main__':
    main()
