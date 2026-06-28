# Running a trillion-parameter model on a Raspberry Pi — the honest, buildable path

This is the engineering answer (not theory) to "run 1T params on a single weak
Pi, local, offline." It separates what is **physics** from what is **work**.

## The one hard limit (physics, proven — do not fight it)

`docs/PROOFS_KOLMOGOROV.md` Theorem 4: lossless **dense** inference must move the
model's full entropy across the slow boundary **per token**. On a Pi's storage
(~0.4–0.9 GB/s) a dense 1T model is **minutes-to-hours per token**. Therefore:

> **Fast + lossless + dense + interactive 1T on a weak Pi is impossible.** No
> software paradigm shift changes this. It is a theorem, not a missing trick.

## What IS achievable today (engineering, real)

**Run 1T offline on a Pi, correctly, slowly** — by streaming the model from disk
layer-by-layer through the Pi's small RAM. This is exactly what AirLLM (this repo)
already does to run 405B on 8GB. The mechanism is demonstrated, on pure CPU, in
`kolmogorov/systems/cpu_streaming_demo.py` (weights live on disk; forward runs on
CPU; correct token out; model never fully resident).

Usable-speed regimes (from `kolmogorov/systems/raspberry_pi_1T.py`):
- **Native MoE + 2-bit**: ~15–60 s/token on a USB SSD — a real offline assistant.
- **Offline batch**: amortize one stream over many prompts → seconds/token-equiv.
- **Dense interactive**: not viable (the physics limit above).

## The concrete build (what a Pi port actually needs)

1. **CPU/ARM streaming path.** AirLLM's core is CUDA-centric (`device="cuda:0"`,
   `torch.cuda.Stream`, pinned memory, BetterTransformer/sdpa). A Pi needs a
   float32 CPU path (fp16 CPU ops are largely unimplemented) with disk→RAM
   streaming and `mmap` from a USB SSD.
2. **Version hardening.** Live demo here hit a `transformers` 5.12 offload
   meta-tensor bug on GPT-NeoX; the Pi path must pin a working
   transformers/accelerate combination (GPT-2 disk-offload works today — see demo).
3. **Sub-layer granularity** so a single 1T layer (which alone can exceed 8GB at
   fp16) is streamed in pieces.
4. **2-bit / native-MoE weights** on the SSD to cut bytes/token (the only speed
   lever; lossy for dense, exact for native MoE).
5. **Batch mode** for offline workloads (the lossless throughput win).

## Bottom line
The dream, stated precisely, splits in two: the *interactive dense lossless*
version is forbidden by Theorem 4; the *run-it-offline* version is **already this
repository's technology** and needs a CPU/ARM streaming path to land on a Pi. That
port is real, finite engineering — not a paradigm shift, and not a fantasy.
