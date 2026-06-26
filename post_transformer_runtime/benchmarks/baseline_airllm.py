"""
AirLLM baseline adapter + CLI.

This is THE baseline every post-transformer experiment must beat (Phase 0
guardrail #1: "compare to real AirLLM, not a fantasy"). It wraps AirLLM's
``AutoModel`` behind the :class:`EngineAdapter` protocol so ``runner.py`` can
measure it on the six axes.

Runtime deps (torch, airllm) are imported lazily inside ``AirLLMAdapter.load``
so this file -- and the rest of the harness -- imports fine on a machine that
only has the standard library (e.g. this CI container, or for ``--help``).

Usage:
    python -m post_transformer_runtime.benchmarks.baseline_airllm \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --device cpu \\
        --max-new-tokens 8 \\
        --out results/qwen05b_cpu.json

Start SMALL (0.5B-3B) where ground truth is reachable; scale only once the
harness itself is trusted (Phase 0 guardrail #5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .runner import run_benchmark
from .report import format_result, result_to_markdown


# A short, fixed, license-clean eval text for perplexity. Keep it deterministic
# and identical across engines so the quality axis is comparable.
DEFAULT_EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models predict the next token given the previous ones. "
    "Memory, computation, storage, and energy all matter for local inference."
)

DEFAULT_PROMPT = "Explain in one sentence what makes local LLM inference hard:"


class AirLLMAdapter:
    """EngineAdapter wrapping AirLLM's AutoModel."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        compression: Optional[str] = None,
        max_seq_len: int = 256,
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.compression = compression
        self.max_seq_len = max_seq_len
        self.hf_token = hf_token
        self.name = "airllm" + (f"-{compression}" if compression else "")
        self.model = None
        self.model_dir: Optional[str] = None
        self.splitted_dir: Optional[str] = None
        self._torch = None

    def load(self) -> float:
        """Construct the AirLLM model; return load time in seconds."""
        t0 = time.time()
        import torch  # noqa: F401  (validates torch presence early)
        from airllm import AutoModel

        self._torch = torch
        kwargs = dict(compression=self.compression, max_seq_len=self.max_seq_len)
        if self.device:
            kwargs["device"] = self.device
        if self.hf_token:
            kwargs["hf_token"] = self.hf_token
        self.model = AutoModel.from_pretrained(self.model_id, **kwargs)

        # AirLLM exposes the original cache path and the splitted path.
        self.model_dir = str(getattr(self.model, "model_local_path", "") or "") or None
        self.splitted_dir = str(getattr(self.model, "checkpoint_path", "") or "") or None
        return time.time() - t0

    # --- EngineAdapter protocol ------------------------------------------- #

    def encode(self, text: str) -> Sequence[int]:
        ids = self.model.tokenizer(text, return_tensors="pt").input_ids
        return ids[0].tolist()

    def generate(self, token_ids: Sequence[int], max_new_tokens: int) -> Sequence[int]:
        torch = self._torch
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long)
        out = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            use_cache=False,
            return_dict_in_generate=False,
        )
        # generate returns full sequence; slice off the prompt.
        full = out[0].tolist() if hasattr(out, "__getitem__") else list(out)
        return full[len(token_ids):]

    def token_logprobs(self, token_ids: Sequence[int]) -> Optional[Sequence[float]]:
        """Teacher-forcing per-token logprobs for perplexity.

        Truncates to max_seq_len to respect AirLLM's fixed attention window.
        """
        torch = self._torch
        ids = list(token_ids)[: self.max_seq_len]
        if len(ids) < 2:
            return None
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.inference_mode():
            out = self.model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out[0]
        # logits: [1, seq, vocab]; predict token t from position t-1.
        logits = logits[0].float()
        logprobs_all = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        picked = logprobs_all[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
        return picked.tolist()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the AirLLM baseline (Post-Transformer Runtime Phase 5).",
    )
    parser.add_argument("--model", required=True, help="HF repo id or local path")
    parser.add_argument("--device", default="cpu", help='e.g. "cpu", "cuda:0"')
    parser.add_argument("--compression", default=None, choices=[None, "4bit", "8bit"],
                        help="optional bitsandbytes compression (CUDA only)")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="tokens to generate for the throughput measurement")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--eval-text", default=DEFAULT_EVAL_TEXT,
                        help="fixed text for the perplexity (quality) axis")
    parser.add_argument("--no-energy", action="store_true",
                        help="disable RAPL/GPU energy measurement")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--out", default=None, help="write JSON result to this path")
    parser.add_argument("--markdown", default=None, help="append a markdown row/section")
    args = parser.parse_args(argv)

    adapter = AirLLMAdapter(
        model_id=args.model,
        device=args.device,
        compression=args.compression,
        max_seq_len=args.max_seq_len,
        hf_token=args.hf_token,
    )

    print(f"[airllm-baseline] loading {args.model} on {args.device} "
          f"(compression={args.compression}) ...", file=sys.stderr)
    try:
        load_seconds = adapter.load()
    except ImportError as exc:
        print(f"ERROR: missing runtime dependency: {exc}\n"
              f"Install with: pip install airllm torch  (and bitsandbytes for "
              f"compression).", file=sys.stderr)
        return 2

    result = run_benchmark(
        adapter,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        eval_text=args.eval_text,
        measure_energy=not args.no_energy,
    )
    result.load_seconds = load_seconds

    print(format_result(result))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"[airllm-baseline] wrote {args.out}", file=sys.stderr)

    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown, "a") as fh:
            fh.write(result_to_markdown(result) + "\n")
        print(f"[airllm-baseline] appended markdown to {args.markdown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
