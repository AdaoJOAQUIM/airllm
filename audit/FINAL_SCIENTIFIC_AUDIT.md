# FINAL SCIENTIFIC AUDIT

Independent lab review of the entire `airllm` fork. Method: Cognitive Audit Engine
(truth/tokens) — recon (size + regex mining + structural map) localised the verdict
to the core; deep reads only there; the rest pruned with reasons. No code was
modified (Phase -1 freeze). Note: **no OpenHands/MiniMax project exists in scope** —
this audits what is actually present.

## Executive summary
The repo is a **published, working memory-offload inference library** (AirLLM) plus
**this session's research prototypes** (research/tme) plus **standard finetuning
scripts**. AirLLM's claims about *memory* hold; its *paradigm* framing does not — it
is a classic space-time tradeoff with standard quantization. The research stack is
honest, self-tested lab work, explicitly not a paradigm shift. The biggest evidence
gap is an in-repo *quality* benchmark for compression and an integration test of the
layered-inference path.

## The nine verdict questions

1. **What works?** AirLLM layered streaming (70B/405B in 4-8GB VRAM) and 4/8-bit
   compression round-trip. research/tme modules all pass `--selftest`.
2. **What is proven (in-repo)?** Compression round-trip RMSE<0.1 (GPU); adapter
   selection; research/tme measured effects within the toy domain (reuse,
   transfer 0→2, MDL 8→7, depth ladder 2→4→5), held-out validated.
3. **What is promising?** Compression's "tiny accuracy loss" (plausible via
   bitsandbytes, not benchmarked here); research/tme as a measurement/abstraction
   substrate *if* bridged to a real proposer+codebase.
4. **What is speculative?** That research/tme amplification transfers to real airllm
   edits (15%) — the LLM-proposer/real-code bridge is unbuilt.
5. **What is false / refuted?** "1 token = ~1e9 actions" / "replaces an LLM" / "99%
   savings everywhere" — false; amplification is variable, proposer-dependent,
   collapses to ~1 on novel work. AirLLM as a *new paradigm* — false (8%).
6. **Directions to continue:** (a) add a perplexity/task benchmark for 4/8-bit to
   convert the quality claim from promising→proven; (b) a CPU integration test of
   the inference path; (c) if pursuing research/tme, build the real proposer bridge
   and re-measure on actual edits with the CEE.
7. **Directions to abandon:** fixed token-equivalence claims; "paradigm shift"
   language for offloading; deep work on the pruned training trees (no novel claim).
8. **Real paradigm shift?** **No.** AirLLM = pragmatic space-time tradeoff. research/
   tme = honest incremental lab work (memoization → abstraction), useful but not a
   new computing paradigm by its own measured results.
9. **Real maturity level:** AirLLM = **production** (shipped, used; modest tests).
   research/tme = **prototype/lab** (deterministic, self-tested, toy domain).
   training/finetune = **standard/secondary**.

## Limits of this audit (honest)
- GPU-gated tests (`.cuda()`) were **not executed** here (no GPU); correctness of
  compression is read+reasoned, not re-run.
- AirLLM end-to-end inference was not run (needs model weights + GPU).
- The verdict is evidence-weighted (see `BELIEF_STATE.md`), not absolute.

## Recommendation
Keep AirLLM framed honestly (memory tradeoff, not paradigm). Close the two test gaps
above. Treat research/tme as a lab substrate; only scale what the CEE proves >1.
