# Capsule: training / rlhf / anima / scripts / eval (pruned zone)

- **Hypothesis:** these trees add novel methods.
- **Evidence (cheap):** `training/qlora.py`, `rlhf/qlora_dpo.py` = standard QLoRA /
  DPO finetuning; `anima_100k/*` = long-context (100k) Llama training + eval
  notebooks; `scripts/`, `eval/` = dataset/eval helpers and notebooks.
- **Counter-evidence:** none needed — no novel-paradigm claim is attached to them.
- **Risk:** spending audit tokens here yields ~0 information for the audit question.
- **Confidence:** 85% these are conventional, correct-enough finetuning artifacts.
- **Decision:** PRUNED (scientific pruning). Secondary; standard techniques. Not
  deep-read by design. Revisit only if a specific claim is made about them.
