"""
Self-test for the benchmark harness, runnable WITHOUT torch / GPU / downloads.

It feeds ``run_benchmark`` a deterministic dummy adapter (a tiny n-gram-ish toy
"model") to prove the harness *mechanics* work: timing, peak-memory sampling,
storage measurement, energy meter, perplexity computation, and reporting.

This validates the measuring instrument, NOT any model. It deliberately reports
the toy's numbers as the toy's -- never as AirLLM's. Run:

    python -m post_transformer_runtime.benchmarks.selftest
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from .runner import run_benchmark
from .report import format_result


class _DummyAdapter:
    """A deterministic toy engine implementing EngineAdapter.

    'Tokenizes' by whitespace into integer ids via a rolling vocab, 'generates'
    by echoing a fixed continuation, and produces uniform-ish logprobs so
    perplexity is a finite, predictable number. It also allocates a small chunk
    of memory and burns a little CPU time so the samplers have something to see.
    """

    name = "dummy-toy"
    model_id = "toy://selftest"
    device = "cpu"
    compression = None

    def __init__(self, work_dir: str):
        self.model_dir = work_dir
        self.splitted_dir = work_dir
        self._vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self._vocab.setdefault(tok, len(self._vocab))

    def encode(self, text: str) -> Sequence[int]:
        return [self._id(t) for t in text.split()]

    def generate(self, token_ids: Sequence[int], max_new_tokens: int) -> Sequence[int]:
        # Simulate per-token cost (AirLLM is slow per token) and some memory use.
        produced = []
        ballast = []
        for i in range(max_new_tokens):
            ballast.append(bytearray(1_000_000))  # ~1MB/token to move peak RSS
            # tiny busy-wait so seconds_per_token is measurably > 0
            t_end = time.time() + 0.01
            while time.time() < t_end:
                _ = math.sqrt(i + 1)
            produced.append((token_ids[-1] if token_ids else 0) + i + 1)
        del ballast
        return produced

    def token_logprobs(self, token_ids: Sequence[int]) -> Optional[Sequence[float]]:
        # Deterministic "model": logprob = -log(vocab_size_seen) -> finite ppl.
        rng = random.Random(1234)
        n = max(2, len(self._vocab))
        return [math.log(1.0 / n) + rng.uniform(-0.01, 0.01)
                for _ in range(len(token_ids) - 1)]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        # Create a couple of files so dir_size_bytes returns something real.
        Path(d, "shard_a.bin").write_bytes(os.urandom(2_000_000))
        Path(d, "shard_b.bin").write_bytes(os.urandom(3_000_000))

        adapter = _DummyAdapter(work_dir=d)
        result = run_benchmark(
            adapter,
            prompt="local llm inference is hard because memory",
            max_new_tokens=5,
            eval_text="the quick brown fox jumps over the lazy dog again and again",
            measure_energy=True,
        )
        result.load_seconds = 0.0
        print(format_result(result))

        # ---- assertions: the instrument must behave ---------------------- #
        assert result.generated_tokens == 5, result.generated_tokens
        assert result.generate_seconds and result.generate_seconds > 0
        assert result.tokens_per_second and result.tokens_per_second > 0
        assert result.seconds_per_token and result.seconds_per_token > 0
        assert result.peak_rss_bytes and result.peak_rss_bytes > 0
        # ~5MB of shards on disk
        assert result.model_disk_bytes and result.model_disk_bytes >= 4_000_000
        assert result.perplexity is not None and math.isfinite(result.perplexity)
        # uniform 1/n model => perplexity ~= vocab size seen
        assert result.perplexity > 1.0
        print("\nSELFTEST OK: harness mechanics verified (no torch needed).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
