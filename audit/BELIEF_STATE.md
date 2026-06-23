# BELIEF_STATE (live)

Confidence = P(claim is true / supported by in-repo evidence). Updated as evidence
arrives. No OpenHands/MiniMax project was found in scope; targets below are what
actually exists in the repo.

| Hypothesis | Confidence | Basis |
|---|---|---|
| AirLLM runs huge models in tiny VRAM (memory claim) | **92%** | mechanism read in `utils`/`base`; standard offload; widely used |
| AirLLM is latency/IO-bound (reloads weights per token) | **88%** | design implies per-layer disk reads each forward |
| 4/8-bit compression round-trip is correct | **95%** | `test_compression` RMSE<0.1 |
| "tiny accuracy loss" end-to-end (quality) | **45%** | not benchmarked in-repo; relies on bitsandbytes' reputation |
| AirLLM is a new computing *paradigm* | **8%** | it is a space-time tradeoff + standard quant, not new primitives |
| research/tme effects real within toy domain | **80%** | passing selftests, held-out validation, honest negatives |
| research/tme transfers to real airllm edits as-is | **15%** | LLM-proposer + real-code bridge not built; toy DSL only |
| training/rlhf/anima contain novel methods | **15%** | standard QLoRA/DPO/long-context (pruned) |
| Repo overall = production library + lab prototypes | **90%** | size+structure+tests recon |

Refuted / low-confidence and STOPPED:
- "1 token = 1e9 actions / replaces an LLM" → **~2%** (established false across the
  prototype work; amplification is variable and proposer-dependent). Pruned.
