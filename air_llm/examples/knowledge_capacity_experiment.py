"""
Knowledge-capacity experiment (miniature of Allen-Zhu & Li,
arXiv:2404.05405), runnable on a CPU in minutes.

Protocol:
1. Build N random facts: key (a, b) -> value v, with v uniform over 256
   classes, so each fact carries exactly 8 bits of irreducible knowledge.
2. Train a small fixed-size network to memorize them. The knowledge it
   stores is lower-bounded by 8 bits per correctly recalled fact (chance
   recall is negligible: 1/256).
3. Sweep N to find the capacity plateau -> bits per parameter.
4. Take a saturated model and quantize its weights to k bits
   (blockwise absmax, k = 8, 4, 3, 2), re-measure recall -> the
   knowledge-destruction curve. Allen-Zhu & Li report capacity surviving
   int8 but collapsing at low bit widths; this reproduces that shape at
   toy scale.

Everything is deterministic (seeded). This is day one of the research
program in docs/ROADMAP_1T.md, not a claim of new laws: the point is a
real, reproducible curve produced by the repo's own bench.
"""

import math

import torch
import torch.nn as nn


torch.manual_seed(0)

N_A, N_B = 128, 128          # key space: (a, b), 16384 possible keys
N_VALUES = 256               # 8 bits of knowledge per fact
DIM, HIDDEN = 16, 64
MAX_STEPS = 4000
LR = 5e-3


class FactNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_emb = nn.Embedding(N_A, DIM)
        self.b_emb = nn.Embedding(N_B, DIM)
        self.fc1 = nn.Linear(2 * DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_VALUES)

    def forward(self, a, b):
        h = torch.cat([self.a_emb(a), self.b_emb(b)], dim=-1)
        return self.fc2(torch.relu(self.fc1(h)))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def make_facts(n, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randperm(N_A * N_B, generator=g)[:n]
    a, b = keys // N_B, keys % N_B
    v = torch.randint(0, N_VALUES, (n,), generator=g)
    return a, b, v


def recall(model, a, b, v):
    with torch.no_grad():
        return (model(a, b).argmax(-1) == v).float().mean().item()


def train(n_facts, seed=1):
    torch.manual_seed(seed)
    model = FactNet()
    a, b, v = make_facts(n_facts, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(MAX_STEPS):
        opt.zero_grad()
        loss = loss_fn(model(a, b), v)
        loss.backward()
        opt.step()
        if step % 200 == 199 and recall(model, a, b, v) == 1.0:
            break
    return model, (a, b, v)


def quantize_blockwise(model, n_bits, blocksize=64):
    """Blockwise absmax uniform quantization of every weight, in place on
    a copy. n_bits=16 means no quantization."""
    import copy
    q_model = copy.deepcopy(model)
    if n_bits >= 16:
        return q_model
    levels = 2 ** (n_bits - 1) - 1  # symmetric
    with torch.no_grad():
        for p in q_model.parameters():
            flat = p.flatten()
            n = flat.numel()
            pad = (-n) % blocksize
            padded = torch.cat([flat, torch.zeros(pad)])
            blocks = padded.reshape(-1, blocksize)
            absmax = blocks.abs().amax(1, keepdim=True).clamp(min=1e-12)
            q = torch.round(blocks / absmax * levels).clamp(-levels, levels)
            deq = (q / levels * absmax).flatten()[:n]
            p.copy_(deq.reshape(p.shape))
    return q_model


def main():
    bits_per_fact = math.log2(N_VALUES)
    model_params = n_params(FactNet())
    print(f"model: {model_params:,} parameters | {bits_per_fact:.0f} bits per fact")
    print(f"\n== capacity sweep (bits stored vs parameters) ==")
    print(f"{'facts':>7} {'recall':>7} {'bits stored':>12} {'bits/param':>11}")

    saturated = None
    for n_facts in (1000, 2000, 4000, 8000, 16000):
        model, (a, b, v) = train(n_facts)
        acc = recall(model, a, b, v)
        stored = acc * n_facts * bits_per_fact
        print(f"{n_facts:>7} {acc:>7.3f} {stored:>12,.0f} {stored / model_params:>11.3f}")
        if n_facts >= 8000:
            saturated = (model, (a, b, v))

    model, (a, b, v) = saturated
    base_acc = recall(model, a, b, v)
    base_bits = base_acc * len(v) * bits_per_fact
    print(f"\n== quantization destruction curve (saturated model) ==")
    print(f"{'weight bits':>11} {'recall':>7} {'knowledge kept':>15}")
    for n_bits in (16, 8, 4, 3, 2):
        q_model = quantize_blockwise(model, n_bits)
        acc = recall(q_model, a, b, v)
        kept = acc * len(v) * bits_per_fact / base_bits
        print(f"{n_bits:>11} {acc:>7.3f} {kept:>14.1%}")


if __name__ == '__main__':
    main()
