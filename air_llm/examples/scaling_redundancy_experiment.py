"""
E12 - the deciding measurement (docs/RESEARCH_CHARTER.md, element 6):
does exploitable redundancy grow with model size?

For each hidden size we lightly train a FactNet on a fixed fact task,
then measure the lossless compressibility (bits/param, repo codec) of
the TRAINED weights vs a RANDOM-init twin. Random weights are
incompressible (~16 bits/param); any drop for trained weights is
exploitable structure. We track how that gap scales.
"""
import sys
sys.path.insert(0, '..')
import torch, torch.nn as nn
from airllm.lossless import compression_report

torch.manual_seed(0)
N_A, N_B, N_VALUES, N_FACTS, STEPS = 64, 64, 64, 800, 150

def make_facts(n, g):
    keys = torch.randperm(N_A*N_B, generator=g)[:n]
    return keys//N_B, keys%N_B, torch.randint(0, N_VALUES, (n,), generator=g)

def build(hidden, dim=16):
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.a=nn.Embedding(N_A,dim); s.b=nn.Embedding(N_B,dim)
            s.f1=nn.Linear(2*dim,hidden); s.f2=nn.Linear(hidden,N_VALUES)
        def forward(s,a,b):
            return s.f2(torch.relu(s.f1(torch.cat([s.a(a),s.b(b)],-1))))
    return Net()

def bits_per_param(model):
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    return compression_report(sd)['bits_per_param']

def train(model):
    g = torch.Generator().manual_seed(1)
    a,b,v = make_facts(N_FACTS, g)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lf = nn.CrossEntropyLoss()
    for _ in range(STEPS):
        opt.zero_grad(); lf(model(a,b), v).backward(); opt.step()
    return model

def main():
    print(f"{'hidden':>7} {'params':>9} {'random b/p':>11} {'trained b/p':>12} {'gap':>6}")
    for hidden in (32, 64, 128, 256):
        torch.manual_seed(7)
        rnd = build(hidden); r = bits_per_param(rnd)
        t = bits_per_param(train(build(hidden)))
        p = sum(x.numel() for x in rnd.parameters())
        print(f"{hidden:>7} {p:>9,} {r:>11.2f} {t:>12.2f} {r-t:>6.2f}")
    print("\ngap = exploitable redundancy (bits/param). Widening with size ->")
    print("support for the compressibility thesis; flat -> against it.")

if __name__ == '__main__':
    main()
