"""Empirical probe for Certified Lazy Inference (lossless).

Tests the central claim: on real text, the next-token argmax *locks early* — the
final predicted token is already determined long before the full computation
finishes. If so, an exact (lossless) early-stopping scheme could read only a
fraction of the model per token on average.

Two measurements on GPT-2 / WikiText-2:
  (1) DEPTH-LOCK via the logit lens: for each position, the smallest layer k such
      that argmax(head(h_k)) == final argmax AND stays equal for all deeper
      layers. lock_depth = k / L is the fraction of the network actually needed.
  (2) MARGIN & ENTROPY of the final distribution (the confidence that a lossless
      certificate would exploit).

This measures the *potential* of the average-case crack; it is not itself the
certificate (that needs a perturbation bound — the open brick). Deterministic,
CPU-friendly.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

MODEL = "gpt2"


@torch.no_grad()
def run(max_chunks=8, seq_len=256):
    tok = GPT2TokenizerFast.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL, output_hidden_states=True)
    model.eval()
    ln_f, head = model.transformer.ln_f, model.lm_head
    L = model.config.n_layer

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    ids = ids[: (max_chunks + 1) * seq_len]

    lock_fracs, margins, entropies = [], [], []
    for c in range(max_chunks):
        chunk = ids[c * seq_len:(c + 1) * seq_len].unsqueeze(0)
        out = model(chunk)
        hs = out.hidden_states                 # tuple length L+1
        final_logits = out.logits[0]           # [T, V]
        final_arg = final_logits.argmax(-1)    # [T]

        # logit lens per layer: argmax of head(ln_f(h_k))
        per_layer_arg = []
        for k in range(L + 1):
            ll = head(ln_f(hs[k][0]))          # [T, V]
            per_layer_arg.append(ll.argmax(-1))
        per_layer_arg = torch.stack(per_layer_arg, 0)   # [L+1, T]

        T = chunk.shape[1]
        for t in range(T):
            col = per_layer_arg[:, t]           # [L+1]
            target = final_arg[t]
            # smallest k such that col[k..L] all == target
            lock = L
            for k in range(L, -1, -1):
                if col[k] == target:
                    lock = k
                else:
                    break
            lock_fracs.append(lock / L)
            top2 = torch.topk(final_logits[t], 2).values
            margins.append((top2[0] - top2[1]).item())
            p = torch.softmax(final_logits[t], -1)
            entropies.append(-(p * (p + 1e-12).log()).sum().item())

    import statistics as st
    n = len(lock_fracs)
    def pct(xs, q): return sorted(xs)[int(q * (len(xs) - 1))]
    res = {
        "model": MODEL, "n_tokens": n, "n_layers": L,
        "lock_depth_frac": {
            "mean": round(st.mean(lock_fracs), 3),
            "median": round(pct(lock_fracs, 0.5), 3),
            "p90": round(pct(lock_fracs, 0.9), 3),
        },
        "frac_locked_by_half_depth": round(sum(1 for x in lock_fracs if x <= 0.5) / n, 3),
        "frac_locked_by_75pct_depth": round(sum(1 for x in lock_fracs if x <= 0.75) / n, 3),
        "margin_logit": {"median": round(pct(margins, 0.5), 3),
                          "p10": round(pct(margins, 0.1), 3)},
        "entropy_nats": {"mean": round(st.mean(entropies), 3),
                         "median": round(pct(entropies, 0.5), 3)},
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--out", default="kolmogorov/report/lazy_probe_results.json")
    args = ap.parse_args()
    res = run(max_chunks=args.max_chunks)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    f = res["frac_locked_by_half_depth"]
    print(f"\n>>> {100*f:.0f}% of tokens have their EXACT final argmax already "
          f"locked by half the network's depth.")
    print(">>> If a cheap certificate could detect this lock, those tokens would "
          "cost ~half the weights, losslessly. That is the crack.")


if __name__ == "__main__":
    main()
