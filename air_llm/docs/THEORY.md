# Theory: theorems, proofs, and concept architecture

Discipline of this project: before each implementation step, state the
theorems that constrain it, the proof (or proof sketch + source), and the
architecture that follows. Code comes last. Each brick below maps to a
module in this repository.

---

## Theorem 1 (Lossless floor) — pigeonhole

**Statement.** No injective (lossless) encoding maps all possible
1T-parameter fp16 weight settings into 12 GB.

**Proof.** There are 2^(16·10^12) distinct weight settings and only
2^(8·12·10^9) distinct 12 GB files; an injection from a larger finite set
into a smaller one cannot exist. ∎

**Empirical floor for *trained* weights:** ~11 bits/param — the exponent
byte of bf16 weights is low-entropy, the mantissa is near-random
([DFloat11, arXiv:2504.11651](https://arxiv.org/abs/2504.11651); ZipNN).

**Brick:** `airllm/lossless.py` (`compression='lossless'`). **Status: DONE,
measured 16.00 → 11.27 bits/param, bit-for-bit exact.**

---

## Theorem 2 (Knowledge capacity) — Allen-Zhu & Li

**Statement.** Transformer LMs store ~2 bits of extractable factual
knowledge per parameter; the capacity survives int8 quantization and is
destroyed below ~4 bits/weight of storage.

**Proof.** Experimental, via controlled synthetic biographies and
information-theoretic counting of recoverable tuples
([arXiv:2404.05405](https://arxiv.org/abs/2404.05405)).

**Corollary.** Sub-2-bit compression of a *given* model must destroy
knowledge — consistent with the practical quantization frontier sitting at
~2 bits/param ([AQLM](https://arxiv.org/abs/2401.06118),
QTIP, [DBF](https://arxiv.org/abs/2505.11076)). Facts are better stored
*outside* weights: text holds ~8 bits/byte, parameters ~1 bit/byte (fp16).

**Brick:** external fact store (retrieval), later step.

---

## Theorem 3 (Streaming working set) — this step

**Statement.** A linear map y = xW^T with W of size R×C (b bytes/element)
can be computed exactly with peak *weight* memory b·k·C for any block size
k ≤ R, plus |x| + |y| activation memory — independent of R.

**Proof.** Partition the rows of W into ⌈R/k⌉ blocks W_i. Then
y[:, i·k:(i+1)·k] = x W_i^T depends only on block i; blocks can be loaded,
used, and freed sequentially. Exactness is trivial (no arithmetic changes;
each output column is produced once). ∎

**Corollary (throughput bound).** Generation speed is bounded by
tokens/s ≤ disk_bandwidth / bytes_touched_per_token. Compression that
shrinks bytes on disk (Theorem 1 brick) raises this bound; caching layers
that fit in RAM raises it further. This is AirLLM's fundamental regime:
the disk is the memory hierarchy's last level, as in classical
out-of-core computation.

**Requirement it discharges:** with 4 GB RAM, a single layer of a large
model (which can exceed RAM by itself) still executes: the working set is
one *block of one tensor*, not one layer.

**Brick:** `airllm/streaming.py` (`TensorStreamer`, `streamed_linear`).
**Status: this commit.**

---

## Theorem 5 (Pipelining bound) — two-stage pipeline

**Statement.** Computing n blocks where each block needs d seconds of I/O
and c seconds of compute takes n·(c+d) sequentially, but only
d + n·max(c, d) with one prefetch thread (load block i+1 while computing
block i). I/O is fully hidden when c ≥ d; the speedup approaches 2× when
c = d.

**Proof.** Induction on i: when compute of block i starts, its data is
already resident (loaded during compute of block i−1, which took
max(c,d) ≥ d). The critical path is the slower stage plus the initial
fill. ∎ (Classical result; same wheel as AirLLM's existing layer-level
prefetch and llama.cpp's overlapped reads.)

**Corollary.** Combined with Theorem 3's corollary, a disk-bound
deployment (c < d) loses *nothing* to compute: generation speed equals the
disk-bandwidth bound itself. Every byte saved by the Theorem 1 codec
converts 1:1 into throughput.

**Brick:** `streaming.py` (`prefetch=True` in `streamed_linear`,
`StreamedLinear` module). **Status: this commit.**

---

## Theorem 6 (Blockwise quantization error bound)

**Statement.** Quantizing a block of B weights by q(w) = a·c(w/a), where
a = max|w| over the block and c(·) maps to the nearest of K codebook
values, has elementwise error |w − q(w)| ≤ a·δ, where δ is half the
largest gap between adjacent codebook values on [−1, 1].

**Proof.** w/a ∈ [−1, 1]; the nearest codebook value is at distance at
most δ; multiplying by a scales the error. ∎ Small blocks keep `a` local,
which is why blockwise beats per-tensor scaling
([Dettmers & Zettlemoyer, arXiv:2212.09720](https://arxiv.org/abs/2212.09720));
the NF4 codebook places the K = 16 values at standard-normal quantiles,
information-theoretically optimal for Gaussian-like trained weights
([QLoRA, arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

**Brick:** `airllm/quant_cpu.py` (`compression='4bit-cpu' / '8bit-cpu'`),
pure CPU — the existing '4bit'/'8bit' modes require bitsandbytes+CUDA.
**Status: done.**

---

## Theorem 4 (Prediction = compression) — Delétang et al.

**Statement.** A predictive model plus arithmetic coding is a lossless
compressor whose code length is the model's log-loss; conversely any
compressor induces a predictor ([ICLR 2024,
arXiv:2309.10668](https://arxiv.org/abs/2309.10668); Shannon 1948;
Solomonoff 1964).

**Consequence.** "A better model in the same byte budget" and "better
compression" are the same objective; the project's single metric is
capability vs. bits/param. The optimal inductor (Solomonoff/AIXI) is
uncomputable (reduction from the halting problem), so all bricks are
approximations — measured, not assumed.

**Brick:** `examples/lossless_bench.py` (storage side). Capability side:
later step.

---

## Concept architecture

```
                    12 GB disk budget
  ┌────────────────────────────────────────────────────┐
  │  reasoner weights        facts (lossless)  programs │
  │  (ternary/low-bit,       text corpus +     library  │
  │   lossless-coded,        local index       of code  │
  │   ~2-5 GB)               (~6-10 GB)        (grows)  │
  └───────┬────────────────────┬──────────────────┬─────┘
          │ Theorem 3:         │ Theorem 2:       │ Kolmogorov:
          │ stream blocks,     │ facts outside    │ procedures as
          │ RAM ≥ one block    │ weights          │ programs
  ┌───────▼────────────────────▼──────────────────▼─────┐
  │            4 GB RAM: activations + one block        │
  │        reasoner streams → retrieves → writes code   │
  └──────────────────────────────────────────────────────┘
   measured by: bits/param bench (Theorem 1) + capability bench (Theorem 4)
```

Execution order: 1. lossless codec (done) → 2. tensor streaming (this
commit) → 3. low-bit CPU decode (BitNet/AQLM-style) → 4. fact store +
program library. See ROADMAP_1T.md for sources per step.
