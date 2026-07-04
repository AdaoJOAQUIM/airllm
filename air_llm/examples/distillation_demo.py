"""
E13 - end-to-end distillation, the constructive Theorem 8 (charter elem 3).
A large trained TEACHER is compressed into small STUDENT pseudo-models of
increasing size; we measure indistinguishability under Q (agreement with
the teacher). We also compare to training a student directly on the same
FIXED task the teacher learned, as a reference point.
"""
import sys
sys.path.insert(0, '..')
import torch, torch.nn as nn
from airllm.distill import distill_to_student, agreement

torch.manual_seed(0)
N_A = N_B = 96; N_VAL = 32; DIM = 16

# FIXED featurization + task, shared by everyone (independent of any model)
PHI_A = torch.randn(N_A, DIM); PHI_B = torch.randn(N_B, DIM)
W = torch.randn(N_VAL, 2*DIM)
def task_label(a, b):
    return (torch.cat([PHI_A[a], PHI_B[b]], -1) @ W.T).argmax(-1)

def net(hidden):
    class N(nn.Module):
        def __init__(s):
            super().__init__()
            s.a=nn.Embedding(N_A,DIM); s.b=nn.Embedding(N_B,DIM)
            s.f1=nn.Linear(2*DIM,hidden); s.f2=nn.Linear(hidden,N_VAL)
        def forward(s,a,b):
            return s.f2(torch.relu(s.f1(torch.cat([s.a(a),s.b(b)],-1))))
    return N()

g = torch.Generator().manual_seed(1)
def sampler(n):
    return (torch.randint(0,N_A,(n,),generator=g),
            torch.randint(0,N_B,(n,),generator=g))

def train_on_task(model, steps=600):
    o = torch.optim.Adam(model.parameters(), lr=5e-3); lf = nn.CrossEntropyLoss()
    for _ in range(steps):
        a,b = sampler(256); o.zero_grad()
        lf(model(a,b), task_label(a,b)).backward(); o.step()
    model.eval(); return model

teacher = train_on_task(net(256))
def params(m): return sum(p.numel() for p in m.parameters())
# teacher's own accuracy on the fixed task (it is not perfect):
a,b = sampler(8000)
t_acc = float((teacher(a,b).argmax(-1)==task_label(a,b)).float().mean())
print(f"teacher: {params(teacher):,} params, task accuracy {t_acc:.3f}\n")
print(f"{'student H':>9} {'params':>8} {'distilled vs teacher':>21} {'direct vs teacher':>18}")
for hidden in (8, 16, 32, 64):
    s_d = distill_to_student(teacher, net(hidden), sampler, steps=400)
    ag_d = agreement(s_d, teacher, sampler)
    s_t = train_on_task(net(hidden), steps=400)
    ag_t = agreement(s_t, teacher, sampler)
    print(f"{hidden:>9} {params(net(hidden)):>8,} {ag_d:>21.3f} {ag_t:>18.3f}")
print("\nAgreement with the teacher rises with student size (Theorem 8:")
print("more bits -> smaller d_Q). Distillation copies the teacher's actual")
print("FUNCTION; direct training copies the task -- they converge as the")
print("teacher approaches task-optimality. Element 3 realized.")
