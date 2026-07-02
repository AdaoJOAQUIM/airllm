"""
Discriminating experiment for the SGD-channel mechanism
(docs/CAPACITY_THEORY.md, section 4).

If knowledge capacity is the capacity of a Gaussian channel
kappa <= 1/2 log2(1 + SNR), then injecting gradient noise must collapse
the stored bits along the log curve. If capacity resists the injected
noise, the channel mechanism is refuted and storage is attractor-like
(pointing to singular learning theory instead).

Protocol: same fact-memorization task as knowledge_capacity_experiment
(N = 6000 facts, near capacity of the 22,848-param net), trained with
Gaussian noise added to every gradient at relative scale sigma
(noise std = sigma * grad std, per tensor, per step).
"""

import math

import torch

from knowledge_capacity_experiment import (FactNet, make_facts, recall,
                                           n_params, MAX_STEPS, LR, N_VALUES)


N_FACTS = 6000


def train_noisy(n_facts, sigma, seed=1):
    torch.manual_seed(seed)
    model = FactNet()
    a, b, v = make_facts(n_facts, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(MAX_STEPS):
        opt.zero_grad()
        loss = loss_fn(model(a, b), v)
        loss.backward()
        if sigma > 0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(sigma * p.grad.std() * torch.randn_like(p.grad))
        opt.step()
        if step % 200 == 199 and recall(model, a, b, v) == 1.0:
            break
    return model, (a, b, v)


def main():
    bits_per_fact = math.log2(N_VALUES)
    params = n_params(FactNet())
    print(f"model: {params:,} params | {N_FACTS} facts x {bits_per_fact:.0f} bits "
          f"= {N_FACTS * bits_per_fact:,.0f} bits offered")
    print(f"\n{'sigma':>6} {'recall':>7} {'bits stored':>12} {'bits/param':>11}")
    for sigma in (0.0, 1.0, 2.0, 4.0):
        model, (a, b, v) = train_noisy(N_FACTS, sigma)
        acc = recall(model, a, b, v)
        stored = acc * N_FACTS * bits_per_fact
        print(f"{sigma:>6.1f} {acc:>7.3f} {stored:>12,.0f} {stored / params:>11.3f}")
    print("\nchannel mechanism predicts bits/param falling with sigma along "
          "1/2 log2(1+SNR); flat = mechanism refuted (attractor storage).")


if __name__ == '__main__':
    main()
