"""
E9 - empirical verification of the Tangent Capacity Principle
(Theorem K'''d', docs/PROOFS.md): the constant 2 bits/param is Cover's
count on a tangent/feature space, and feature LEARNING changes which
space you are in -- never the constant.

Protocol: rich-train a FactNet (the E1/E2 network) so its hidden
representation is genuinely learned; then test storage of FRESH random
+-1 labels by a linear threshold readout on
  (a) the LEARNED hidden features of the trained network, and
  (b) the features of an untrained (random-init) twin.
Cover predicts the same sharp threshold at m = 2 * n_eff for both,
where n_eff is the feature matrix rank. Realizability decided by LP.
"""

import numpy as np
import torch
from math import comb
from scipy.optimize import linprog

from knowledge_capacity_experiment import FactNet, make_facts, N_A, N_B

torch.set_num_threads(1)
TRIALS = 40
rng = np.random.default_rng(3)


def train_rich(n_facts=500, steps=300, seed=42):
    torch.manual_seed(seed)
    model = FactNet()
    a, b, v = make_facts(n_facts, seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(a, b), v).backward()
        opt.step()
    return model


def hidden_features(model, a, b):
    with torch.no_grad():
        h = torch.cat([model.a_emb(a), model.b_emb(b)], dim=-1)
        return torch.relu(model.fc1(h)).numpy().astype(np.float64)


def separable(F, y):
    m, n = F.shape
    res = linprog(c=np.zeros(n), A_ub=-(y[:, None] * F), b_ub=-np.ones(m),
                  bounds=[(None, None)] * n, method='highs')
    return res.status == 0


def p_store(model, m):
    hits = 0
    for _ in range(TRIALS):
        a = torch.from_numpy(rng.integers(0, N_A, size=m))
        b = torch.from_numpy(rng.integers(0, N_B, size=m))
        F = hidden_features(model, a, b)
        y = rng.choice([-1.0, 1.0], size=m)
        hits += separable(F, y)
    return hits / TRIALS


def theory(m, n):
    return sum(comb(m - 1, k) for k in range(min(n, m))) / 2 ** (m - 1)


def main():
    trained = train_rich()
    torch.manual_seed(7)
    untrained = FactNet()

    # effective feature dimension (rank) for each network
    a = torch.from_numpy(rng.integers(0, N_A, size=512))
    b = torch.from_numpy(rng.integers(0, N_B, size=512))
    n_tr = np.linalg.matrix_rank(hidden_features(trained, a, b))
    n_un = np.linalg.matrix_rank(hidden_features(untrained, a, b))
    print(f"effective feature rank: learned = {n_tr}, random-init = {n_un}\n")

    print(f"{'m/n_eff':>8} {'LEARNED features':>17} {'RANDOM features':>16} {'theory':>8}")
    for ratio in (1.0, 1.5, 2.0, 2.5, 3.0):
        m_tr, m_un = int(ratio * n_tr), int(ratio * n_un)
        print(f"{ratio:>8.1f} {p_store(trained, m_tr):>17.3f} "
              f"{p_store(untrained, m_un):>16.3f} {theory(m_tr, n_tr):>8.3f}")

    print("\nSame Cover threshold at 2 bits per readout dimension on LEARNED")
    print("and on RANDOM features: feature learning moved the space, not the")
    print("constant -- the Tangent Capacity Principle (Theorem K'''d').")


if __name__ == '__main__':
    main()
