"""
E4 - empirical verification of Theorem K' (docs/PROOFS.md):
the perceptron stores random labels up to EXACTLY 2 bits per parameter,
with a sharp threshold at m = 2n and P_store(2n, n) = 1/2.

For each ratio m/n we draw m Gaussian points in R^n (general position
a.s.), i.i.d. random +-1 labels, and test realizability -- a linear
feasibility problem y_i (w . x_i) >= 1, decided exactly by LP
(Corollary K'.3: storage and efficient search coexist here).

The theory column is the closed form of Lemma K'.2:
P_store = Pr[ Bin(m-1, 1/2) <= n-1 ].
"""

import numpy as np
from math import comb
from scipy.optimize import linprog


N = 24          # parameters
TRIALS = 60
rng = np.random.default_rng(0)


def separable(X, y):
    # feasibility: -y_i * (x_i . w) <= -1 ; minimize 0
    m, n = X.shape
    A_ub = -(y[:, None] * X)
    b_ub = -np.ones(m)
    res = linprog(c=np.zeros(n), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * n, method='highs')
    return res.status == 0


def theory(m, n):
    # Lemma K'.2: P_store = 2^(1-m) * sum_{k<=n-1} binom(m-1, k)
    return sum(comb(m - 1, k) for k in range(min(n, m))) / 2 ** (m - 1)


def main():
    print(f"n = {N} parameters, {TRIALS} trials per point; "
          f"theory: P = Pr[Bin(m-1,1/2) <= n-1]\n")
    print(f"{'m/n':>5} {'m':>4} {'P_store measured':>17} {'P_store theory':>15}")
    for ratio in (1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0):
        m = int(ratio * N)
        hits = 0
        for _ in range(TRIALS):
            X = rng.normal(size=(m, N))
            y = rng.choice([-1.0, 1.0], size=m)
            hits += separable(X, y)
        print(f"{ratio:>5.1f} {m:>4} {hits / TRIALS:>17.3f} {theory(m, N):>15.3f}")
    print(f"\nSharp threshold at m/n = 2 with P = 1/2 exactly at the "
          f"critical point:\nkappa_perceptron = 2 bits/parameter (Theorem K').")


if __name__ == '__main__':
    main()
