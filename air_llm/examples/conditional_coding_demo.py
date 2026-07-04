"""
E10 - Theorem 9 (docs/SHANNON.md): the Shannon floor H(X) drops to the
conditional floor H(X|S) once a shared model S is available. We measure
both on a structured source with the repo's own CTW coder (induction.py):
  - marginal (order-0) entropy  ~ H(X)      : Shannon's naive floor
  - CTW code length             ~ H(X|past) : the shared-model floor
The gap is the legal circumvention of Shannon -- no bound is broken;
the naive statement just used the wrong entropy.
"""
import math, sys
sys.path.insert(0, '..')
from airllm.induction import CTWPredictor

def order0_bits_per_symbol(bits):
    n = len(bits); ones = sum(bits); p = ones / n
    if p in (0.0, 1.0): return 0.0
    return -(p*math.log2(p) + (1-p)*math.log2(1-p))

def ctw_bits_per_symbol(bits, depth=12):
    p = CTWPredictor(depth=depth); p.update_bits(bits)
    return p.logloss_bits() / len(bits)

def main():
    # structured source: period-5 pattern with 2% random flips
    import random; rng = random.Random(0)
    pat = [0,1,1,0,1]
    bits = [pat[i%5]^(1 if rng.random()<0.02 else 0) for i in range(4000)]
    h0 = order0_bits_per_symbol(bits)
    hc = ctw_bits_per_symbol(bits)
    print(f"marginal H(X)      (Shannon naive floor): {h0:.3f} bits/symbol")
    print(f"conditional H(X|S) (shared CTW model):    {hc:.3f} bits/symbol")
    print(f"circumvention factor: {h0/hc:.1f}x  -- same data, wrong vs right entropy")
    print("Shannon intact: H(X|S) <= H(X), conditioning reduces entropy (Thm 9).")

if __name__ == '__main__':
    main()
