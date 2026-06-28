"""Stage 0 experiment (plan §6/§11.2): first b_eff vs Δppl curve.

Probes sub-claim C1 with the weakest generator (G1, low-rank SVD) on GPT-2
small. For a sweep of rank fractions, replaces every large 2-D block weight by
its low-rank approximation, measures held-out WikiText-2 perplexity, and reports
the effective bits/weight. The output is the first real data point on whether
trained weights carry far less information than their nominal size.

Pre-registered (see registry/stage0.yaml): model, seed, eval slice, thresholds.
Deterministic. CPU-only.
"""
from __future__ import annotations

import argparse
import copy
import json

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from kolmogorov.eval.perplexity import compute_perplexity
from kolmogorov.generators.lowrank import (
    low_rank_approx, factor_elements, effective_bits_per_weight,
)

MODEL_NAME = "gpt2"
SEED = 0


def is_target_matrix(name: str, p: torch.Tensor) -> bool:
    """Large 2-D block weights only; skip embeddings, lm_head, LayerNorm, biases."""
    if p.dim() != 2:
        return False
    if "wte" in name or "wpe" in name or "lm_head" in name:
        return False
    if min(p.shape) < 64:
        return False
    return name.endswith(".weight")


def apply_low_rank(model, rank_fraction: float):
    """Replace target weights in-place by low-rank approx; return cost stats."""
    stored, original = 0, 0
    sd = model.state_dict()
    for name, p in model.named_parameters():
        if not is_target_matrix(name, p):
            continue
        d, k = p.shape
        rank = max(1, int(round(rank_fraction * min(d, k))))
        with torch.no_grad():
            p.copy_(low_rank_approx(p.data, rank))
        stored += factor_elements((d, k), rank)
        original += d * k
    # global effective bits/weight over the *compressed* matrices only
    b_eff = 16.0 * stored / original if original else float("nan")
    return b_eff, stored, original


def run(max_chunks: int = 40, seq_len: int = 512,
        fractions=(0.9, 0.5, 0.25, 0.1, 0.05, 0.02)):
    torch.manual_seed(SEED)

    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    base.eval()

    # datasets>=3 requires a namespaced repo id for wikitext; try mirrors.
    ds = None
    for repo in ("Salesforce/wikitext", "mindchain/wikitext2", "wikitext"):
        try:
            cfg = "wikitext-2-raw-v1" if repo != "mindchain/wikitext2" else None
            ds = (load_dataset(repo, cfg, split="test") if cfg
                  else load_dataset(repo, split="test"))
            print(f"loaded eval corpus from {repo}")
            break
        except Exception as e:  # noqa: BLE001 - we genuinely want any fallback
            print(f"  dataset {repo} failed: {type(e).__name__}")
    if ds is None:
        raise RuntimeError("could not load any WikiText mirror")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]

    # truncate corpus to what we evaluate, for speed + reproducibility
    ids = ids[: (max_chunks + 1) * seq_len]

    rows = []
    base_ppl = compute_perplexity(base, ids, seq_len=seq_len, max_chunks=max_chunks)
    rows.append({"fraction": "full(fp16-nominal)", "b_eff": 16.0,
                 "ppl": round(base_ppl, 4), "dppl": 0.0})
    print(f"baseline GPT-2 perplexity (WikiText-2, {max_chunks} chunks): "
          f"{base_ppl:.4f}")

    # CONTROL (harness validation): low-rank at FULL rank must reproduce the
    # baseline. If this is not ~equal to baseline, the harness is broken and no
    # downstream number is trustworthy.
    control = copy.deepcopy(base)
    apply_low_rank(control, 1.0)
    control_ppl = compute_perplexity(control, ids, seq_len=seq_len,
                                     max_chunks=max_chunks)
    assert abs(control_ppl - base_ppl) < 1e-3, (
        f"HARNESS INVALID: full-rank roundtrip {control_ppl} != baseline {base_ppl}")
    rows.append({"fraction": "control(full-rank SVD)", "b_eff": 16.0,
                 "ppl": round(control_ppl, 4), "dppl": round(control_ppl - base_ppl, 6)})
    print(f"control full-rank SVD roundtrip: {control_ppl:.4f}  (== baseline -> "
          f"harness valid)")
    del control

    for f in fractions:
        if f >= 1.0:
            continue
        model = copy.deepcopy(base)
        b_eff, stored, original = apply_low_rank(model, f)
        ppl = compute_perplexity(model, ids, seq_len=seq_len, max_chunks=max_chunks)
        dppl = ppl - base_ppl
        rows.append({"fraction": f, "b_eff": round(b_eff, 3),
                     "ppl": round(ppl, 4), "dppl": round(dppl, 4)})
        print(f"  rank_frac={f:<5} b_eff={b_eff:5.2f} bits/w  "
              f"ppl={ppl:8.3f}  Δppl={dppl:+8.3f}")
        del model

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", type=str, default="kolmogorov/report/stage0_results.json")
    args = ap.parse_args()

    rows = run(max_chunks=args.max_chunks, seq_len=args.seq_len)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("\nInterpretation guide (pre-registered, plan §7/§8):")
    print(" - Naive low-rank is the WEAKEST generator. Any flat-Δppl point below")
    print("   16 bits/w is already compression; below the ~2-4 bits/w quantization")
    print("   frontier at Δppl<=0.2 would be the C1 signal worth chasing with")
    print("   stronger generators. A steep Δppl cliff means structure isn't plain")
    print("   low-rank -> pivot generator family, not abandon C1.")


if __name__ == "__main__":
    main()
