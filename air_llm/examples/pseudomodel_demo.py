"""
E3 - empirical verification of Theorem 8 / formula F2
(docs/INDISTINGUISHABILITY.md).

The decisive control separating the two walls:

- INCOMPRESSIBLE teacher: f is a uniformly random function X -> {0,1}.
  By F2, B* is linear in the domain; a bounded student CANNOT reach
  ε-indistinguishability with few bits. Pigeonhole holds: agreement on
  unseen queries stays at chance (0.5) no matter how we compress. This
  is why "12 GB cannot fake a random 1T function" is a THEOREM.

- STRUCTURED teacher: f is a linear threshold (halfspace) on the input
  bits. It lies in a class of small pseudo-dimension, so by F2 a tiny
  student becomes ε-indistinguishable under Q after few samples --
  Door 2 in action. This is why distillation of *real* (structured)
  models works.

Same student capacity in both cases. The ONLY difference is the teacher's
structure under Q. Runtime: seconds (pure numpy).
"""

import numpy as np


N_BITS = 14                     # domain size D = 2^14 = 16384
D = 1 << N_BITS
rng = np.random.default_rng(0)


def bits_matrix(xs):
    return ((xs[:, None] >> np.arange(N_BITS)[::-1]) & 1).astype(np.float64)


# ---- two teachers, same input domain ----
def random_teacher():
    table = rng.integers(0, 2, size=D)         # incompressible: D random bits
    return lambda xs: table[xs]


def halfspace_teacher():
    w = rng.normal(size=N_BITS)
    bias = 0.0
    return lambda xs: (bits_matrix(xs) @ w + bias > 0).astype(np.int64)


def train_student(teacher, m):
    """Fixed-capacity student = logistic regression on the N_BITS features
    (|S| ~ N_BITS * 32 bits, constant, independent of the teacher)."""
    xtr = rng.integers(0, D, size=m)
    Xtr, ytr = bits_matrix(xtr), teacher(xtr).astype(np.float64)
    w = np.zeros(N_BITS)
    b = 0.0
    lr = 0.5
    for _ in range(300):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - ytr
        w -= lr * (Xtr.T @ g / m + 1e-4 * w)
        b -= lr * g.mean()
    return lambda xs: (1.0 / (1.0 + np.exp(-(bits_matrix(xs) @ w + b))) > 0.5).astype(np.int64)


def agreement_under_Q(student, teacher, n=8000):
    xq = rng.integers(0, D, size=n)            # Q = uniform on the domain
    return float((student(xq) == teacher(xq)).mean())


def main():
    print(f"domain D = 2^{N_BITS} = {D};  student capacity fixed "
          f"(~{N_BITS * 32} bits) in BOTH columns\n")
    print(f"{'samples m':>10} | {'random teacher':>16} | {'halfspace teacher':>18}")
    print(f"{'':>10} | {'d_Q agreement':>16} | {'d_Q agreement':>18}")
    print("-" * 52)
    rand_f = random_teacher()
    half_f = halfspace_teacher()
    for m in (100, 500, 2000, 8000, 20000):
        a_rand = agreement_under_Q(train_student(rand_f, m), rand_f)
        a_half = agreement_under_Q(train_student(half_f, m), half_f)
        print(f"{m:>10} | {a_rand:>16.3f} | {a_half:>18.3f}")
    print("\nrandom teacher -> agreement stuck at chance (0.5): pigeonhole")
    print("  holds, 12 GB cannot fake a random 1T function (F2 lower bound).")
    print("halfspace teacher -> agreement -> 1.0 with a TINY student:")
    print("  structure under Q makes it e-indistinguishable (Door 2).")
    print("Same student both sides. The teacher's structure is the whole story.")


if __name__ == '__main__':
    main()
