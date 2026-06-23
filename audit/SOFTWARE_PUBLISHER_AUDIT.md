# SOFTWARE PUBLISHER AUDIT (A→Z, all angles)

Due-diligence review of the whole repository as a software-publishing org would run
it (security, supply-chain, legal/IP, packaging/release, QA/CI, code quality, docs,
privacy). Token-efficient: batched scans + high-signal file reads; severity-rated.

Severity: **CRITICAL / HIGH / MEDIUM / LOW / INFO**. Threat model: an OSS inference
library that **downloads and executes third-party models** from the Hugging Face Hub.

## Executive verdict
The upstream **AirLLM** library is functional and popular but **not enterprise
release-ready as-is**: it ships three HIGH-risk behaviours typical of the ML-OSS
ecosystem (remote code execution surfaces + an unsafe install hook), a **license
metadata defect**, and **no CI/automated QA**. The fork's own additions
(`research/` + `audit/`) are clean and additive (see `FORK_DELTA_AUDIT.md`). None of
the findings are *malicious*; they are quality/security/compliance gaps.

## Findings

| ID | Area | Sev | Finding | Evidence | Fix |
|----|------|-----|---------|----------|-----|
| SEC-1 | Security | **HIGH** | `trust_remote_code=True` hard-coded in config/tokenizer/model loads → arbitrary code from a model repo runs on the host | `airllm_base.py:121,174,190…`, `auto_model.py:23`, `airllm_baichuan.py`, `airllm_llama_mlx.py` | make it opt-in (default False), warn in docs |
| SEC-2 | Security | **HIGH** | `torch.load(...)` on `.bin` checkpoints = pickle → RCE if the checkpoint is untrusted/poisoned | `utils.py:296,310` | prefer safetensors; gate `.bin` behind explicit opt-in / `weights_only=True` |
| SUP-1 | Supply-chain / packaging | **HIGH** | `setup.py` PostInstall runs `pip install --upgrade transformers` at install → network + mutates the user's env, non-reproducible, can break other packages | `setup.py:7-13` | remove; express as a version constraint instead |
| LEG-1 | Legal / IP | **MEDIUM** | License **mismatch**: PyPI classifier says *MIT*, but `LICENSE` and `funding.json` say *Apache-2.0* | `setup.py:45` vs `LICENSE:1`, `funding.json:26` | set classifier to Apache-2.0; add SPDX headers |
| SUP-2 | Dependencies | **MEDIUM** | Non-reproducible / aged deps: `transformers @ git+main` (unpinned), `peft@v0.3.0`, `accelerate@v0.20.3`, `bitsandbytes==0.39.0`; `setup.py install_requires` has **no version bounds**; `requirements.txt` is actually the *Anima training* stack, not the package's | `requirements.txt`, `setup.py:29-39` | pin/bound versions; separate lib vs training reqs; run `pip-audit` |
| QA-1 | CI / process | **MEDIUM** | `.github/` contains **only** `FUNDING.yml` — no CI, no test automation, no `SECURITY.md`, no Dependabot/CODEOWNERS | `.github/FUNDING.yml` (sole file) | add CI (lint+test), security policy, dependency bot |
| QA-2 | Testing | **MEDIUM** | Only 2 unit tests; `test_compression` is **GPU-only** (`.cuda()`) so it won't run in standard CI; the core layered-inference path has **no test** | `air_llm/tests/*` | add a CPU smoke test of the inference path |
| DOC-1 | Docs / trust | **LOW** | README ships affiliate/`?ref=` links (`bloome.im/login?ref=…`, `godmodeai.co`, `crazyfaceai.com`); reputation = **unknown** (not malicious; `bloome.im` unreachable now) | README "AI Agents Recommendation" | label as ads / move out of README |
| PRIV-1 | Privacy | **LOW/INFO** | `wandb` telemetry in the (training) requirements; many tracking badges in README. The **airllm package itself has no telemetry** | `requirements.txt:9` | none required; note for downstreams |
| QUAL-1 | Code quality | **LOW** | Commented-out code, broad `except Exception: pass` (in `clean_memory`), per-model wrapper duplication, version skew (pkg 2.11.0 vs unrelated requirements.txt) | `utils.py`, `airllm_base.py`, model adapters | tidy; not blocking |
| RES-1 | Fork additions | **INFO** | `research/` + `audit/` (this session): stdlib-only, no network/exec/secrets, self-tested; additive only | `FORK_DELTA_AUDIT.md` | none |

## Area notes
- **Security:** the two RCE surfaces (SEC-1/2) are *inherent to "download any model and
  run it"*, but a publisher ships them **off by default + documented**, not silent.
- **Reproducibility:** SUP-1/2 mean two installs can yield different dependency trees
  — unacceptable for a shipped product; fine-ish for a research repo.
- **Legal:** Apache-2.0 is the intended license (LICENSE + funding.json); the MIT
  classifier is a packaging bug, not a relicense — but it misstates terms on PyPI.
- **QA:** no CI + GPU-only tests = the test suite effectively never runs in
  automation; correctness rests on manual/notebook checks and community usage.

## Release-readiness checklist (publisher gate)
- [ ] SEC-1, SEC-2, SUP-1 remediated (the HIGH items) — **blocking**
- [ ] LEG-1 license metadata corrected — **blocking for distribution**
- [ ] SUP-2 deps pinned/bounded + `pip-audit` clean
- [ ] QA-1 CI with lint+test; QA-2 a CPU smoke test
- [ ] DOC-1 affiliate links labelled/relocated

## Limits of this audit (honest)
- No network here → **no live CVE/`pip-audit` scan**; SUP-2 is surface-level (aged/
  unpinned), not confirmed CVEs.
- GPU-gated tests **not executed** (no GPU); correctness read+reasoned, not re-run.
- Dynamic behaviours (install hook, remote-code loading) assessed by code reading.
