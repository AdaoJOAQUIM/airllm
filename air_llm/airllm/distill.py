"""
End-to-end distillation: Transformer/teacher -> pseudo-model student
(RESEARCH_CHARTER.md element 3; the constructive side of Theorem 8).

distill_to_student() fits a student to a teacher's outputs on samples
drawn from a query distribution Q, which is exactly the algorithm behind
the ε-indistinguishability guarantee (INDISTINGUISHABILITY.md): the
student that agrees with the teacher on enough Q-samples is provably
ε-indistinguishable under Q. Pure torch, CPU-friendly.
"""

import torch
import torch.nn as nn


def distill_to_student(teacher, student, sampler, steps=400, lr=5e-3,
                       temperature=2.0, batch=256, device="cpu"):
    """
    Fit `student` to match `teacher` on inputs from `sampler` (a callable
    n -> batch of teacher-forward args) via soft-label KL distillation.

    teacher, student: callables returning logits over the same label set.
    sampler(n): returns a tuple of tensors to splat into both models.
    Returns the trained student.
    """
    teacher.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    kl = nn.KLDivLoss(reduction="batchmean", log_target=False)
    T = temperature
    for _ in range(steps):
        args = sampler(batch)
        with torch.no_grad():
            t_logits = teacher(*args) / T
            t_prob = torch.softmax(t_logits, dim=-1)
        s_logprob = torch.log_softmax(student(*args) / T, dim=-1)
        loss = kl(s_logprob, t_prob) * (T * T)
        opt.zero_grad()
        loss.backward()
        opt.step()
    student.eval()
    return student


@torch.no_grad()
def agreement(model_a, model_b, sampler, n=8000):
    """d_Q agreement = Pr_{x~Q}[argmax a(x) == argmax b(x)] (1 - distinguishing
    advantage, INDISTINGUISHABILITY.md)."""
    args = sampler(n)
    return float((model_a(*args).argmax(-1) == model_b(*args).argmax(-1))
                 .float().mean())
