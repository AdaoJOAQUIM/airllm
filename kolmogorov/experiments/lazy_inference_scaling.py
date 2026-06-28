"""Does the lossless lazy-inference crack OPEN with scale? (tests Theorem L2)

L2 says expected lossless cost is governed by model confidence (margin/entropy).
Stage 1 showed perplexity falls with scale -> models get more confident. So the
*depth-lock oracle* cost E[C]/full should DECREASE with N if the duality holds.

Measures, across the Pythia suite, the logit-lens depth-lock fraction (the best
possible depth-axis early stop) + margin + entropy. Architecture-generic
(GPT-NeoX / GPT-2). Deterministic, CPU-friendly.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

PYTHIA = {"70m": "EleutherAI/pythia-70m", "160m": "EleutherAI/pythia-160m",
          "410m": "EleutherAI/pythia-410m", "1.4b": "EleutherAI/pythia-1.4b"}
PARAMS = {"70m": 70e6, "160m": 160e6, "410m": 410e6, "1.4b": 1.4e9}


def heads_for(model):
    """Return (final_norm, lm_head, n_layers) across supported architectures."""
    if hasattr(model, "gpt_neox"):          # Pythia / GPT-NeoX
        return model.gpt_neox.final_layer_norm, model.embed_out, model.config.num_hidden_layers
    if hasattr(model, "transformer"):       # GPT-2
        return model.transformer.ln_f, model.lm_head, model.config.n_layer
    raise ValueError("unsupported architecture")


@torch.no_grad()
def probe(size, max_chunks=6, seq_len=256):
    name = PYTHIA[size]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, output_hidden_states=True,
                                                 dtype=torch.float32)
    model.eval()
    final_norm, head, L = heads_for(model)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    ids = ids[: (max_chunks + 1) * seq_len]

    locks, margins, ents = [], [], []
    for c in range(max_chunks):
        chunk = ids[c * seq_len:(c + 1) * seq_len].unsqueeze(0)
        out = model(chunk)
        hs = out.hidden_states
        final_arg = out.logits[0].argmax(-1)
        per_layer = torch.stack([head(final_norm(hs[k][0])).argmax(-1)
                                 for k in range(L + 1)], 0)   # [L+1, T]
        for t in range(chunk.shape[1]):
            col = per_layer[:, t]
            tgt = final_arg[t]
            lock = L
            for k in range(L, -1, -1):
                if col[k] == tgt:
                    lock = k
                else:
                    break
            locks.append(lock / L)
            top2 = torch.topk(out.logits[0, t], 2).values
            margins.append((top2[0] - top2[1]).item())
            p = torch.softmax(out.logits[0, t], -1)
            ents.append(-(p * (p + 1e-12).log()).sum().item())

    n = len(locks)
    return {
        "size": size, "params": PARAMS[size], "n_layers": L, "n_tokens": n,
        "mean_lock_depth_frac": round(st.mean(locks), 3),       # ~ E[C]/full (depth oracle)
        "frac_locked_by_half": round(sum(x <= 0.5 for x in locks) / n, 3),
        "median_margin": round(sorted(margins)[n // 2], 3),
        "mean_entropy_nats": round(st.mean(ents), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["70m", "160m", "410m"])
    ap.add_argument("--max-chunks", type=int, default=6)
    ap.add_argument("--out", default="kolmogorov/report/lazy_scaling_results.json")
    args = ap.parse_args()

    import os
    rows = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {r["size"] for r in rows}
    for s in args.sizes:
        if s in done:
            print(f"[{s}] done, skip"); continue
        r = probe(s, max_chunks=args.max_chunks)
        print(json.dumps(r))
        rows.append(r); rows.sort(key=lambda x: x["params"])
        json.dump(rows, open(args.out, "w"), indent=2)

    print("\n=== LAZY-INFERENCE DUALITY ACROSS SCALE ===")
    print(f"{'N':>10} {'lock_depth':>11} {'<=half':>8} {'margin':>8} {'entropy':>8}")
    for r in sorted(rows, key=lambda x: x["params"]):
        print(f"{r['params']:>10.0f} {r['mean_lock_depth_frac']:>11} "
              f"{r['frac_locked_by_half']:>8} {r['median_margin']:>8} "
              f"{r['mean_entropy_nats']:>8}")
    print("\nIf mean_lock_depth_frac DECREASES with N -> the lossless crack opens "
          "with scale (L2 duality holds). If flat -> it stays closed.")


if __name__ == "__main__":
    main()
