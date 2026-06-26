# Post-Transformer Runtime — Benchmark Harness (Phase 5 baseline)

This is the **measuring instrument** for the whole project. Before any "axis"
(sparsity, quantization, hypernetworks, …) can claim a win, it must be measured
here against the **real AirLLM baseline**, on the **same machine and model**.

> Phase 0 guardrail (see `docs/AIRLLM_AUTOPSY.md`): *no victory without a
> reproducible benchmark; a method that saves memory but destroys quality has
> proven nothing.* This harness exists to make that rule enforceable.

## What it measures (the six axes)

| Axis | Field(s) | How |
|---|---|---|
| Host memory | `peak_rss_bytes` | background RSS sampler (psutil → `/proc` → `getrusage`) |
| Device memory | `peak_gpu_bytes` | torch peak counter → `nvidia-smi` |
| Storage | `model_disk_bytes`, `splitted_disk_bytes` | recursive dir size |
| Throughput | `tokens_per_second`, `seconds_per_token` | timed generation on a fixed prompt |
| Energy | `cpu_energy_joules`, `gpu_energy_joules`, `joules_per_token` | Intel RAPL + integrated `nvidia-smi` power |
| Quality | `perplexity` | teacher-forcing log-likelihood on a fixed eval text |

**Honesty contract:** any quantity that cannot be measured on the current
machine is reported as `null`/`n/a` — *never* guessed. No NVIDIA GPU → VRAM and
GPU-energy are `n/a`. No Intel RAPL → CPU energy is `n/a`.

## Files

| File | Role |
|---|---|
| `metrics.py` | measurement primitives + `BenchmarkResult` (stdlib-only fallbacks) |
| `runner.py` | engine-agnostic `run_benchmark()` + `EngineAdapter` protocol |
| `baseline_airllm.py` | AirLLM adapter + CLI (the reference baseline) |
| `report.py` | text / markdown formatting |
| `selftest.py` | validates the harness mechanics **without torch/GPU/downloads** |

## Quick start

Verify the instrument itself (no dependencies beyond the stdlib):

```bash
python -m post_transformer_runtime.benchmarks.selftest
```

Benchmark the AirLLM baseline on a **small** model (start where ground truth is
reachable — Phase 0 guardrail #5):

```bash
pip install airllm torch          # + bitsandbytes for --compression on CUDA

python -m post_transformer_runtime.benchmarks.baseline_airllm \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device cpu \
    --max-new-tokens 8 \
    --out results/qwen0_5b_cpu.json
```

On a GPU box, compare fp16 vs 4bit to see the storage/throughput trade-off the
autopsy describes (§1.4):

```bash
python -m post_transformer_runtime.benchmarks.baseline_airllm \
    --model meta-llama/Meta-Llama-3-8B --device cuda:0 \
    --max-new-tokens 8 --out results/llama8b_fp16.json
python -m post_transformer_runtime.benchmarks.baseline_airllm \
    --model meta-llama/Meta-Llama-3-8B --device cuda:0 --compression 4bit \
    --max-new-tokens 8 --out results/llama8b_4bit.json
```

## Adding a future engine

Implement the `EngineAdapter` protocol (`name`, `model_id`, `device`,
`encode`, `generate`, `token_logprobs`, and optional `model_dir` /
`splitted_dir` / `compression`) and pass an instance to `run_benchmark`. It is
then measured on the exact same axes as AirLLM, so the Phase 5 comparison is
apples-to-apples by construction.

## A note on expectations

The autopsy derives (analytically, to be confirmed here) that AirLLM is
disk-bandwidth bound: `tokens/s ≈ disk_bandwidth / model_size`, i.e. **seconds
to minutes per token** for large models, and worse on a Raspberry Pi. Keep
`--max-new-tokens` small for big models, or you will wait a long time. That
slowness is not a harness bug — it is the baseline reality this project aims to
break.

## Results policy

The `results/` directory holds JSON/markdown produced **on real hardware**.
Committed numbers must state the host they were measured on (every JSON embeds
a `host` block). Do **not** commit hand-written or estimated numbers as results.
