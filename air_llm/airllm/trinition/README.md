# Trinition operators (experimental sandbox)

A small, **dependency-free, pure-Python** sandbox for experimenting numerically
with "Trinition" hypercomplex algebra and Atangana-style fractional operators.

## What this is

Concrete, runnable, *testable* implementations of three ideas:

| Concept | File | What it actually does |
|---|---|---|
| **Trinition number** `Z = a + b·i + c·j` | `trinition.py` | A 3D hypercomplex value over a *configurable* multiplication table. Commutativity, non-commutativity and non-associativity are measurable via `commutator()` / `associator()`, and a deformation parameter `alpha` lets you dial them continuously. |
| **Fractional derivative** | `operators.py` | A correct Grünwald–Letnikov discrete fractional derivative `D^α` over a Trinition sequence. `α=1` is an ordinary backward difference; `0<α<1` is a long-memory operator. |
| **Atangana memory integral** | `operators.py` | A fading-memory accumulator with hard **resets** — old context can be cleared instead of accumulating forever (the "anti-KV-cache" idea, as an operator you can run). |

## What this is **not**

- It is **not** part of AirLLM's inference path and changes nothing about how
  AirLLM runs models. It pulls in no third-party dependencies.
- It is **not** a validated replacement for attention, matrices, or
  transformers, and it makes **no** performance claim. The thesis that "LLMs
  compute intelligence in the wrong algebra" is an unproven hypothesis; this
  code is a place to explore the building blocks, not evidence for the claim.
- The specific multiplication table is **one configurable choice**, not a
  canonical "Atangana table." A closed 3D algebra has no unique product, which
  is exactly why the table is explicit and parametrized.

## Quick start

```python
from airllm.trinition import (
    Trinition, make_structure_constants, from_vector,
    fractional_derivative, atangana_memory,
)

i = Trinition(0, 1, 0)
j = Trinition(0, 0, 1)
print(i * i)            # Trinition(-1 + 0i + 0j)   (i^2 = -1 by default)
print(i.commutator(j)) # non-zero -> non-commutative

# Continuously deform from commutative (alpha=0) to non-commutative (alpha=1):
sym = make_structure_constants(alpha=0.0)
print(Trinition(0,1,0,sym).commutator(Trinition(0,0,1,sym)).norm())  # ~0

# Fractional derivative of a sequence of Trinition values:
seq = [from_vector([t, 0, 0]) for t in range(6)]
print([round(z.a, 3) for z in fractional_derivative(seq, alpha=0.5)])

# Resettable memory (reset clears history at the given indices):
ones = [from_vector([1, 0, 0]) for _ in range(6)]
print([round(z.a, 3) for z in atangana_memory(ones, retention=0.5, resets=[3])])
```

## Run the demo and tests

```bash
# Demo (from the airllm package dir, to avoid importing the heavy AirLLM stack):
cd air_llm/airllm && python -c "import trinition.demo as d; d.main()"

# In a full AirLLM install (torch/tqdm present) this also works:
python -m airllm.trinition.demo

# Tests (pure stdlib unittest, no dependencies):
cd air_llm && python -m unittest tests.test_trinition
```

## Falsification harness

`benchmark.py` puts the hypothesis to an actual test: it compares composition
algebras (real / quaternion / Trinition) at **equal readout-parameter budgets**
on a rotation-composition control task and a fractional-memory task, and sweeps
the Trinition deformation knob. It requires `numpy`.

```bash
cd air_llm/airllm && python -c "import trinition.benchmark as b; b.main(['--seed','0'])"
```

The recorded outcome (real numbers, including the parts that contradict the
thesis) is in [`RESULTS.md`](RESULTS.md): the deformable Trinition product does
**not** capture rotation structure (no alpha approaches the matched algebra),
while the fractional operator shows a modest, genuine parameter-efficiency edge
on long-memory data.

`learn_algebra.py` goes further and *learns the entire multiplication table* by
gradient descent at dimensions 3 and 4 — the strongest form of the deformability
claim. It shows the real obstacle is **dimension, not tuning**: a learned 3D
bilinear product plateaus ~5× worse than a learned 4D one (which approaches the
exact quaternion answer), because SO(3) composition is bilinear only in 4D.

```bash
cd air_llm/airllm && python -c "import trinition.learn_algebra as la; la.main(['--seed','0'])"
```

## The surviving signal: fractional memory vs. a state-space baseline

`benchmark_longmemory.py` is the decisive test for the one positive result —
does a *learnable* fractional order hold up against a diagonal state-space model
(the S4/S4D core), not just a naive AR window? It does, in a scoped way: a
single fractional-order parameter matches a ~3× larger SSM on polynomial
long-memory data and loses on exponential-memory data (an honest cross-over).

```bash
cd air_llm/airllm && python -c "import trinition.benchmark_longmemory as bm; bm.main(['--seed','0'])"
```

[`THEORY.md`](THEORY.md) proves the underlying separation (`O(1)` fractional
params vs `Θ(log L)` SSM modes for power-law memory, with the regime where it
fails), and [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) lays out the path to a
publishable result — including an honest gating checklist that says where to quit.

## Anomalous diffusion: where the algebra is literal physics

`anomalous_diffusion.py` is the one setting where "the right algebra" is not a
metaphor — it is the governing law of anomalous transport. A single fractional
parameter recovers the physical Hurst exponent of fractional-Brownian-motion
trajectories (mean error ≈ 0.055) and matches a 10-parameter AR model. This is
the only result here with a path to the *Nature family*.

```bash
cd air_llm/airllm && python -c "import trinition.anomalous_diffusion as ad; ad.main(['--seed','0'])"
```

[`NATURE_PATH.md`](NATURE_PATH.md) is the straight answer about *Nature*
specifically: why the ML results are NeurIPS-shaped not Nature-shaped, why
anomalous diffusion is the genuine route (Nature Communications / Physics /
Methods, via the AnDi challenge), and the hard 90% — real data, beating the
AnDi state of the art, a finding about a real system, domain co-authors — that
cannot be done in this sandbox.

`andi_eval.py` runs the gate on the **official AnDi benchmark** (`andi_datasets`,
*Nature Communications* 2021), Task 1, across all five diffusion models. Result:
the 1-parameter fractional estimator matches the classical TA-MSD baseline on
fBM/SBM (Gaussian long memory) and is mis-specified on CTRW/ATTM/LW — a clean
regime map, not a state-of-the-art win (see `RESULTS.md`).

```bash
pip install andi-datasets
cd air_llm/airllm && python -c "import trinition.andi_eval as ae; ae.main(['--seed','0'])"
```

## Notes / further reading

The naming follows discussions of Abdon Atangana's work on 3D hypercomplex
("Trinition") numbers and fractional calculus (e.g. the Atangana–Baleanu
derivative). Those are real areas of mathematics; the application to language
models here is purely exploratory.
