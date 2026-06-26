"""
Generic benchmark runner.

The runner is deliberately decoupled from AirLLM: it drives anything that
satisfies a tiny ``EngineAdapter`` protocol (load / tokenize / generate /
logits). AirLLM is one adapter (``baseline_airllm.py``); future post-transformer
engines plug in the same way and are measured on the exact same axes, so the
comparison in Phase 5 is apples-to-apples.

What is measured (all via metrics.py, all None-able):
  * load_seconds                          time to construct/prepare the engine
  * generate_seconds / tokens_per_second  throughput on a fixed prompt
  * peak_rss_bytes / peak_gpu_bytes       peak host & device memory
  * model_disk_bytes / splitted_disk_bytes on-disk footprint
  * cpu/gpu energy (joules) + per token   when RAPL / nvidia-smi exist
  * perplexity                            quality, on a fixed eval text

Nothing here invents a number: if an adapter can't provide logits, perplexity
stays None; if there is no GPU, the GPU fields stay None.
"""

from __future__ import annotations

import time
import math
from typing import Optional, Protocol, runtime_checkable, Sequence

from .metrics import (
    BenchmarkResult,
    HostInfo,
    PeakMemorySampler,
    EnergyMeter,
    dir_size_bytes,
)


@runtime_checkable
class EngineAdapter(Protocol):
    """Minimal interface a benchmarkable engine must expose."""

    #: human-readable engine name, e.g. "airllm" or "airllm-4bit"
    name: str
    #: model identifier (repo id or path)
    model_id: str
    #: device string, e.g. "cuda:0" / "cpu"
    device: str
    #: optional: directory whose size = original on-disk model
    model_dir: Optional[str]
    #: optional: directory whose size = splitted/streamed model
    splitted_dir: Optional[str]
    #: optional: compression tag, e.g. "4bit"
    compression: Optional[str]

    def encode(self, text: str) -> Sequence[int]:
        """Return token ids for ``text``."""

    def generate(self, token_ids: Sequence[int], max_new_tokens: int) -> Sequence[int]:
        """Generate and return the produced token ids (new tokens only)."""

    def token_logprobs(self, token_ids: Sequence[int]) -> Optional[Sequence[float]]:
        """Per-token log p(token_t | token_<t) for perplexity, or None.

        Returns a list of length len(token_ids)-1 (one logprob per predicted
        token). Adapters that cannot expose teacher-forcing logits return None.
        """


def _perplexity(logprobs: Sequence[float]) -> Optional[float]:
    if not logprobs:
        return None
    mean_nll = -sum(logprobs) / len(logprobs)
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return float("inf")


def run_benchmark(
    adapter: EngineAdapter,
    prompt: str,
    max_new_tokens: int = 8,
    eval_text: Optional[str] = None,
    measure_energy: bool = True,
) -> BenchmarkResult:
    """Drive ``adapter`` through load / generate / quality phases under
    measurement and return a populated :class:`BenchmarkResult`.

    The adapter is expected to be *already constructed* (so that load time is
    measured by the caller around construction if desired); here we re-time a
    lightweight ``encode`` warmup as a sanity check and focus on generation,
    memory, storage, energy and quality.
    """
    result = BenchmarkResult(
        engine=getattr(adapter, "name", "unknown"),
        model_id=getattr(adapter, "model_id", None),
        device=getattr(adapter, "device", None),
        compression=getattr(adapter, "compression", None),
        host=HostInfo.collect().to_dict(),
    )

    # --- storage footprint (cheap, do first) ------------------------------- #
    model_dir = getattr(adapter, "model_dir", None)
    splitted_dir = getattr(adapter, "splitted_dir", None)
    if model_dir:
        result.model_disk_bytes = dir_size_bytes(model_dir)
    if splitted_dir:
        result.splitted_disk_bytes = dir_size_bytes(splitted_dir)

    # --- tokenize prompt --------------------------------------------------- #
    prompt_ids = list(adapter.encode(prompt))
    result.prompt_tokens = len(prompt_ids)

    # --- generation under memory + energy measurement --------------------- #
    sampler = PeakMemorySampler()
    energy = EnergyMeter() if measure_energy else None

    t0 = time.time()
    with sampler:
        if energy is not None:
            with energy:
                produced = list(adapter.generate(prompt_ids, max_new_tokens))
        else:
            produced = list(adapter.generate(prompt_ids, max_new_tokens))
    gen_seconds = time.time() - t0

    result.generated_tokens = len(produced)
    result.generate_seconds = gen_seconds
    result.peak_rss_bytes = sampler.peak_rss_bytes
    result.peak_gpu_bytes = sampler.peak_gpu_bytes

    if produced:
        result.seconds_per_token = gen_seconds / len(produced)
        if gen_seconds > 0:
            result.tokens_per_second = len(produced) / gen_seconds

    if energy is not None:
        result.cpu_energy_joules = energy.cpu_energy_joules
        result.gpu_energy_joules = energy.gpu_energy_joules
        total_e = sum(
            e for e in (energy.cpu_energy_joules, energy.gpu_energy_joules)
            if e is not None
        )
        if produced and (energy.cpu_energy_joules is not None
                         or energy.gpu_energy_joules is not None):
            result.joules_per_token = total_e / len(produced)

    # --- quality: perplexity on a fixed eval text ------------------------- #
    if eval_text:
        eval_ids = list(adapter.encode(eval_text))
        result.eval_text_tokens = len(eval_ids)
        try:
            logprobs = adapter.token_logprobs(eval_ids)
        except Exception as exc:  # never let quality break the run
            logprobs = None
            result.notes.append(f"perplexity skipped: {exc!r}")
        if logprobs is not None:
            result.perplexity = _perplexity(list(logprobs))
        else:
            result.notes.append("perplexity unavailable for this adapter")

    return result
