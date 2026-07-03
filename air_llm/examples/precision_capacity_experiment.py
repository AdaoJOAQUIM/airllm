"""
E7 - empirical verification of Theorem K''' (a)-(c) (docs/PROOFS.md):
storage capacity is a property of usable PRECISION, not of parameter
count -- so no representational constant "2 bits/param" exists, and
robustness caps capacity at kappa <= log2(c/rho) bits per weight.

One single float64 parameter w stores K bits b_1..b_K as the binary
expansion w = sum b_i 2^-i; extraction is the doubling map
(bit = [x >= 1/2]; x <- 2x - bit). We measure:
  1. how many bits ONE parameter recalls exactly (expect ~50, machine
     precision) -- kappa >> 2 representationally, refuting the naive
     universal constant;
  2. recalled bits under perturbation w + U(-rho, rho) versus the proven
     robustness line log2(1/rho) (Theorem K'''c).
"""

import math
import random


def encode(bits):
    return sum(b * 2.0 ** -(i + 1) for i, b in enumerate(bits))


def decode(w, k):
    out, x = [], w
    for _ in range(k):
        bit = 1 if x >= 0.5 else 0
        out.append(bit)
        x = 2.0 * x - bit
    return out


def prefix_correct(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    rng = random.Random(0)
    K = 60
    trials = 200

    # 1. noiseless capacity of ONE parameter
    exact = min(prefix_correct(decode(encode(bits := [rng.randint(0, 1) for _ in range(K)]), K), bits)
                for _ in range(trials))
    print(f"one float64 parameter, no noise: {exact} bits recalled exactly")
    print(f"  -> kappa(representational) ~ {exact} bits/param >> 2 : the naive")
    print(f"     universal constant is refuted (Theorem K'''b); capacity is precision.\n")

    # 2. the robustness line kappa(rho) <= log2(c/rho)
    print(f"{'rho':>10} {'bits recalled (mean)':>21} {'log2(1/rho)':>12}")
    for rho in (1e-2, 1e-4, 1e-6, 1e-9, 1e-12):
        total = 0
        for _ in range(trials):
            bits = [rng.randint(0, 1) for _ in range(K)]
            w = encode(bits) + rng.uniform(-rho, rho)
            total += prefix_correct(decode(w, K), bits)
        mean = total / trials
        print(f"{rho:>10.0e} {mean:>21.1f} {math.log2(1 / rho):>12.1f}")
    print("\nRecalled bits track log2(1/rho) (Theorem K'''c, measured):")
    print("robustness, not architecture, is what bounds bits per parameter --")
    print("the empirical '2 bits/param' of trained nets must be a property of")
    print("the TRAINING DYNAMICS' effective noise floor (K'''d, open).")


if __name__ == '__main__':
    main()
