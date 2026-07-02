# Roadmap: trillion-class capability on weak hardware

Goal: get the most capability possible out of a machine like a Raspberry Pi
(4 GB RAM, ~12 GB disk), fully offline — and track the shortest honest path
toward "1T-class" capability on such hardware.

## What theory allows (and forbids)

- **Bit-for-bit lossless storage of arbitrary 1T-parameter weights in 12 GB is
  impossible** — a counting/pigeonhole argument, and experimentally the lossless
  floor for bf16 weights sits at ~11 bits/param
  ([DFloat11, arXiv:2504.11651](https://arxiv.org/abs/2504.11651)).
- **LLMs store ~2 bits of knowledge per parameter**
  ([Allen-Zhu & Li, arXiv:2404.05405](https://arxiv.org/abs/2404.05405)); the
  practical near-lossless quantization frontier is also ~2 bits/param
  ([AQLM, arXiv:2401.06118](https://arxiv.org/abs/2401.06118);
  QTIP; [DBF, arXiv:2505.11076](https://arxiv.org/abs/2505.11076)). These two
  numbers meeting is the sign that sub-2-bit compression of a *given* model must
  destroy knowledge.
- **Capability per parameter grows exponentially over time** — density doubles
  every ~3.5 months ([Densing Law, arXiv:2412.04315](https://arxiv.org/abs/2412.04315));
  compute needed for fixed performance halves every ~8 months
  ([Epoch AI, arXiv:2403.05812](https://arxiv.org/abs/2403.05812)). The winning
  strategy is therefore to build the *harness* that every future model flows
  through, not to over-optimize any single model.
- **Prediction = compression** ([Delétang et al., arXiv:2309.10668](https://arxiv.org/abs/2309.10668)):
  a better model in the same byte budget IS better compression. Knowledge is
  cheaper outside the weights: raw text stores facts ~30x more densely than
  parameters, and programs are denser still (Kolmogorov).

## The plan, in order of increasing ambition

1. **Lossless shard codec — DONE** (`airllm/lossless.py`, `compression='lossless'`).
   Byte-plane splitting + entropy coding, pure CPU, no bitsandbytes/CUDA.
   Measured: 16.00 → 11.27 bits/param (−29.6%) on trained-like bf16 weights,
   bit-for-bit exact, ~140 MB/s decompression on one CPU core — decode is faster
   than the disk read it saves. Matches the DFloat11/ZipNN numbers.
2. **CPU/ARM-first streaming — PRIMITIVES DONE** (`airllm/streaming.py`).
   `TensorStreamer` reads shard tensors one at a time (mmap-backed lazy
   `safe_open`, lossless shards decoded transparently); `streamed_linear`
   computes y = xW^T exactly in row blocks so peak weight memory is one block
   regardless of the matrix size (proof: docs/THEORY.md, Theorem 3), with
   prefetch of block N+1 during compute of block N (Theorem 5) and a
   `StreamedLinear` nn.Module drop-in whose weights never leave the disk.
   Remaining: wire StreamedLinear into the layer-by-layer forward pass of
   airllm_base.
3. **Low-bit CPU decode paths — 4/8-BIT DONE** (`airllm/quant_cpu.py`,
   `compression='4bit-cpu' / '8bit-cpu'`): blockwise absmax int8 and NF4
   quantization, pure CPU, no bitsandbytes/CUDA, verified end-to-end on a real
   model. Remaining: 2-bit codebook formats (AQLM/QTIP-style) and native
   ternary (BitNet b1.58, 1.58 bits/param, CPU-friendly by construction —
   see [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) and
   [bitnet.cpp](https://github.com/microsoft/BitNet)). A ~60B ternary model is
   the largest knowledge store that physically fits in 12 GB under the
   2-bits/param capacity law.
4. **Externalize knowledge — FACT STORE DONE** (`airllm/factstore.py`):
   zlib-compressed passages + local BM25 retrieval, pure standard library,
   offline; `density_report()` quantifies the ~10-30x storage advantage of
   facts-on-disk over facts-in-weights. Weights should hold the *reasoner*,
   not the facts. First program-induction brick done too
   (`airllm/induction.py`): computable approximations of Solomonoff
   induction — CTW exact Bayesian mixture (Willems 1995), Levin-ordered MDL
   program search over a bounded DSL, compression-based similarity and
   prediction (docs/THEORY.md, Theorem 7). Remaining, long-term: a growing,
   self-refactoring program library
   ([DreamCoder, arXiv:2006.08381](https://arxiv.org/abs/2006.08381);
   [CompressARC, arXiv:2512.06104](https://arxiv.org/abs/2512.06104)).

## The metric

Everything above is judged on one curve: **capability vs. bits/parameter**
(and tokens/s on target hardware). `examples/lossless_bench.py` measures the
storage side; run it on real shards:

```bash
python examples/lossless_bench.py /path/to/model-00001-of-000xx.safetensors
```
