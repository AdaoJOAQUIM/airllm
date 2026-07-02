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
2. **CPU/ARM-first streaming.** Tensor-granularity sharding (a single layer of a
   large model can exceed 4 GB), mmap'd reads, prefetch of tensor N+1 during
   compute of tensor N. This turns AirLLM into the layer-1 engine for
   weak-hardware inference.
3. **Low-bit CPU decode paths.** 2-bit codebook formats (AQLM/QTIP-style) and
   native ternary (BitNet b1.58, 1.58 bits/param, CPU-friendly by construction —
   see [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) and
   [bitnet.cpp](https://github.com/microsoft/BitNet)). A ~60B ternary model is
   the largest knowledge store that physically fits in 12 GB under the
   2-bits/param capacity law.
4. **Externalize knowledge.** Weights should hold the *reasoner*, not the facts:
   local retrieval over a compressed corpus (lossless, verbatim) + a small
   model. Long-term direction: knowledge as programs
   ([DreamCoder, arXiv:2006.08381](https://arxiv.org/abs/2006.08381);
   [CompressARC, arXiv:2512.06104](https://arxiv.org/abs/2512.06104)).

## The metric

Everything above is judged on one curve: **capability vs. bits/parameter**
(and tokens/s on target hardware). `examples/lossless_bench.py` measures the
storage side; run it on real shards:

```bash
python examples/lossless_bench.py /path/to/model-00001-of-000xx.safetensors
```
