# COGNITIVE AUDIT ENGINE

A token-budgeted strategy for scientific audit. Goal optimised: **truth discovered
/ tokens consumed** — never lines read. This document is the engine (Phase -3/-2);
the artifacts it produces live beside it (`PROJECT_MAP.md`, `BELIEF_STATE.md`,
`AUDIT_CACHE/knowledge_capsules/`, `FINAL_SCIENTIFIC_AUDIT.md`).

Auditor stance: independent lab reviewer — not the project's developer or defender.
Experimental truth is prioritised; unproven claims are labelled, not assumed.

## The cognitive budget (meta-layer)

Before reading code, decide *what not to read*. The engine spends tokens only where
they reduce uncertainty. Concretely it runs the cheap, high-information passes first
(size ranking, regex mining, structural map) to localise the few zones that carry
the verdict, then reads only those.

## The nine components (and how each was applied to THIS repo)

1. **Semantic Compression Layer** — never keep raw analysis. Every finding is a
   *knowledge capsule* (50-200 tokens): Hypothesis / Evidence / Counter-evidence /
   Risk / Confidence / Decision. Stored in `AUDIT_CACHE/knowledge_capsules/`.

2. **Regex Mining Engine** — before manual reading, grep for stub/risk markers
   (`TODO FIXME pass mock placeholder NotImplemented random fake`) and domain terms
   (`compress decompress encode decode virtual cache memory weight generation
   attention expert`). *Result here:* 676 domain hits cluster in `air_llm/airllm/*`
   and notebooks; stub/fake markers are almost entirely in **notebooks** and one
   **intentionally-labelled** synthetic stub (`research/.../adaptive_compute_probe.py`).
   The shipped `air_llm` package is near stub-free.

3. **AST / Structural Analysis** — map modules/deps before detail. *Result:* core =
   `airllm_base.py` (642) + `utils.py` (403); per-model files are thin adapters;
   `persist/` is the IO layer. See `PROJECT_MAP.md`.

4. **Information-Gain Reading** — read only high-density code (`compress_layer_state_dict`,
   `split_and_save_layers`, `load_layer`, forward loop). Skip logging/boilerplate.
   Every read must answer: *which uncertainty does this remove?*

5. **Delta Reading** — `FILE_STATE_CACHE` keys files by hash; identical hash → SKIP.
   (For a one-shot audit this prevents re-reading; it pays off across sessions.)

6. **Multi-Level Knowledge (L0-L3)** — always start at L0 (one sentence). Descend
   only when a decision needs it. Capsules carry L0/L1; deep code reads are L2/L3.

7. **Audit Knowledge Graph** — Concept → Module → Code → Hypothesis → Result, rather
   than a file list. E.g. *Layered streaming* → `utils.split_and_save_layers` →
   "70B on 4GB" → validated by design, latency-bound (see capsules).

8. **Scientific Pruning** — when a line is refuted or low-value, STOP and document
   why. *Applied:* the `training/ rlhf/ anima_100k/ scripts/ eval/` trees were
   pruned to "secondary — standard QLoRA/DPO/long-context, no novel claim" without
   deep reading (information gain too low for the audit question).

9. **Belief-State Tracking** — `BELIEF_STATE.md` holds every hypothesis with a
   confidence %; each new evidence updates it. This is the audit's live state.

## Phase pruning for this repo (honesty over ceremony)

The spec lists Phases 0-8 with one file each. The supreme rule forbids maximising
files. This repo is small (~8.1k LoC) and largely known: a published library + a set
of research prototypes + standard training scripts. So phases collapse:

- Phases 0/1 (map, autopsy) → `PROJECT_MAP.md`.
- Phases 2/3/4 (hypotheses, SOTA, red-team) → capsules + `BELIEF_STATE.md`.
- Phases 5/6 (code/benchmark audit) → capsules (findings) + noted test gaps.
- Phase 8 + Verdict → `FINAL_SCIENTIFIC_AUDIT.md`.

Phase 7 (TECHNICAL_PLAN) is produced only *if* a change is later requested — none is
now (Phase -1 freeze: audit only, no code edits).

## Supreme rule

Maximise **truth / tokens**. Not a code reader — a cognitive audit system.
