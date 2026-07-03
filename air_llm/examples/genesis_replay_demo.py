"""
E8 - empirical verification of Theorem P'''a (docs/PROOFS.md):
the shortest description of a trained model is its GENESIS
(code, data reference, seed) -- and replaying it reproduces the weights
bit for bit. Under white-box/genesis access, worst-case efficient P is
TRUE for every efficiently-created target; the crypto wall (P'' iii)
binds query-only black boxes, never published recipes.

Caveat, stated honestly: bit-exact replay assumes the same environment
(library versions, hardware kernels); strictly, the genesis description
includes the environment pin. Within one environment, determinism holds.
"""

import os

import torch

from knowledge_capacity_experiment import FactNet, make_facts

torch.set_num_threads(1)

N_FACTS, STEPS, SEED = 500, 300, 42


def genesis_train():
    torch.manual_seed(SEED)
    model = FactNet()
    a, b, v = make_facts(N_FACTS, SEED)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(STEPS):
        opt.zero_grad()
        loss_fn(model(a, b), v).backward()
        opt.step()
    return model


def weights_blob(model):
    return b''.join(p.detach().contiguous().view(torch.uint8).numpy().tobytes()
                    for p in model.parameters())


def main():
    blob1 = weights_blob(genesis_train())
    blob2 = weights_blob(genesis_train())

    identical = blob1 == blob2
    weight_bytes = len(blob1)
    genesis_bytes = os.path.getsize(__file__) + os.path.getsize(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'knowledge_capacity_experiment.py')) + 8  # + seed

    print(f"replay #1 vs replay #2 bit-for-bit identical: {identical}")
    print(f"weights:             {weight_bytes:,} bytes")
    print(f"genesis description: {genesis_bytes:,} bytes (two scripts + seed)")
    print(f"compression via genesis: {weight_bytes / genesis_bytes:.1f}x here;")
    print(f"for a 1T-param model trained on public data the same argument gives")
    print(f"~10^5-10^6x -- LCDL(genesis access) is tiny (Theorem P'''a).")
    print(f"What a 4GB/12GB offline device lacks is not learnability but the")
    print(f"genesis-replay RESOURCES (data + compute): that gap is the")
    print(f"tradeoff curve LCDL(C), and distillation lives in its interior.")
    if not identical:
        raise SystemExit("ERROR: environment is not deterministic; genesis "
                         "description must pin the environment.")


if __name__ == '__main__':
    main()
