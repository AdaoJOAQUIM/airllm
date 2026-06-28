"""Stage 1 — the decisive scaling law (plan §6, PROOFS Cor 3).

For each model size N in the Pythia suite (same architecture & data, varying
width/depth), sweep per-channel quantization bit-widths, measure WikiText-2
perplexity, and record the *critical bits/weight* b*(N) = smallest bit-width with
Δppl <= tau. b*(N) is an operational estimate of the metric-entropy density
H_eps(N)/n_params.

Decisive question (pre-registered, registry/stage1.yaml):
  Does b*(N) DECREASE with N?
   - decreasing  -> H_eps grows sublinearly -> RD-COMP says streaming inference
                    gets *cheaper per parameter* with scale: the C1 signal.
   - flat/rising -> H_eps = Theta(N) -> the pincer makes disk-streaming
                    fundamentally Omega(model) per token-batch: a clean
                    impossibility, honestly reported.

Deterministic, CPU-friendly. Each model evaluated independently so the run can
be resumed/segmented.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from kolmogorov.eval.perplexity import compute_perplexity
from kolmogorov.generators.quantize import quantize_per_channel, is_quantizable

PYTHIA = {
    "70m":  "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m",
    "410m": "EleutherAI/pythia-410m",
    "1.4b": "EleutherAI/pythia-1.4b",
}
PARAMS = {"70m": 70e6, "160m": 160e6, "410m": 410e6, "1.4b": 1.4e9}
SEED = 0
TAU = 0.5  # pre-registered Δppl threshold defining "critical bits"


def load_corpus(tok, seq_len, max_chunks):
    ds = None
    for repo, cfg in (("Salesforce/wikitext", "wikitext-2-raw-v1"),
                      ("mindchain/wikitext2", None)):
        try:
            ds = (load_dataset(repo, cfg, split="test") if cfg
                  else load_dataset(repo, split="test"))
            break
        except Exception as e:  # noqa: BLE001
            print(f"  dataset {repo} failed: {type(e).__name__}")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids[: (max_chunks + 1) * seq_len]


def eval_model_at_bits(base, ids, seq_len, max_chunks, bits):
    if bits >= 16:
        return compute_perplexity(base, ids, seq_len, max_chunks)
    model = copy.deepcopy(base)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if is_quantizable(name, p):
                p.copy_(quantize_per_channel(p.data, bits))
    ppl = compute_perplexity(model, ids, seq_len, max_chunks)
    del model
    gc.collect()
    return ppl


def run_one(size, seq_len=512, max_chunks=20, bit_widths=(8, 4, 3, 2)):
    torch.manual_seed(SEED)
    name = PYTHIA[size]
    tok = AutoTokenizer.from_pretrained(name)
    base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    base.eval()
    ids = load_corpus(tok, seq_len, max_chunks)

    base_ppl = compute_perplexity(base, ids, seq_len, max_chunks)
    row = {"size": size, "params": PARAMS[size], "base_ppl": round(base_ppl, 4),
           "points": [{"bits": 16, "ppl": round(base_ppl, 4), "dppl": 0.0}]}
    print(f"[{size}] N={PARAMS[size]:.0f}  baseline ppl={base_ppl:.3f}")

    crit = None
    for b in bit_widths:
        ppl = eval_model_at_bits(base, ids, seq_len, max_chunks, b)
        dppl = ppl - base_ppl
        row["points"].append({"bits": b, "ppl": round(ppl, 4), "dppl": round(dppl, 4)})
        print(f"    {b}-bit: ppl={ppl:9.3f}  Δppl={dppl:+8.3f}")
        if dppl <= TAU:
            crit = b  # smallest passing bit-width seen so far (sweep descends)
    row["critical_bits"] = crit
    print(f"    -> critical bits b*({size}) = {crit}  (Δppl<= {TAU})")
    del base
    gc.collect()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["70m", "160m", "410m"])
    ap.add_argument("--max-chunks", type=int, default=20)
    ap.add_argument("--out", default="kolmogorov/report/stage1_results.json")
    args = ap.parse_args()

    prev = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh)
    done = {r["size"] for r in prev}

    rows = list(prev)
    for size in args.sizes:
        if size in done:
            print(f"[{size}] already done, skipping")
            continue
        rows.append(run_one(size, max_chunks=args.max_chunks))
        rows.sort(key=lambda r: r["params"])
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)

    print("\n=== SCALING LAW: critical bits/weight vs N ===")
    for r in sorted(rows, key=lambda r: r["params"]):
        print(f"  N={r['params']:>10.0f}  ({r['size']:>4})  "
              f"b*={r['critical_bits']}  base_ppl={r['base_ppl']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
