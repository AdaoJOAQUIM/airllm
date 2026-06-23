# FORK_DELTA_AUDIT — what this fork committed onto airllm

Token-efficient audit of the **fork's own delta**, not the whole repo. Method: one
path-scoped diff localised the entire change set; targeted scans covered the rest.

## Scope (measured)
- Delta = `75436d1..HEAD` (upstream tip → branch head) = **30 files, +3509 / -1**.
- **100% additive, confined to `research/` + `audit/` + `.gitignore`.**
- **Upstream library code is UNTOUCHED**: path-scoped diff over `air_llm/`,
  `anima_100k/`, `rlhf/`, `training/`, `scripts/`, `eval/` = **empty**.

## Authorship finding
Every non-session commit is by the **upstream maintainer** (`lyo.gavin@gmail.com`,
Yu Li / Gavin Li) + upstream PR contributors. There are **no separate user-authored
code commits**; the only commits beyond upstream are this session's (Claude,
user-directed) research/audit additions.

## All-angles review of the delta

| Angle | Result |
|---|---|
| **Supply-chain / integrity** | ✅ upstream code unmodified; additive only |
| **Secrets / credentials** | ✅ none (scanned: api key/secret/password/ghp_/sk-/AKIA/PRIVATE KEY) |
| **Dangerous calls** | ✅ none (no network/socket/urllib/requests, no subprocess/os.system, no eval/exec/pickle) |
| **I/O footprint** | ✅ writes only a **gitignored** memory JSON (`.engine_memory.json`), optional `runs.jsonl`, and a tempfile in the stress test; nothing written into the library or outside `research/tme/` |
| **Dependencies** | ✅ Python **stdlib only**; no new third-party deps, no requirements changes |
| **Licensing** | ✅ additive docs/code, no third-party code copied; consistent with repo Apache-2.0 (no per-file headers — optional) |
| **Repo hygiene** | ✅ `.gitignore` excludes memory JSON + `__pycache__`; synthetic data is labelled (`sample_runs.synthetic.jsonl`) |
| **Correctness / honesty** | ✅ every `research/**` module passes `--selftest`; claims are held-out validated and labelled (synthetic stubs marked NOT-results; multipliers stated as variable). See capsules. |
| **Privacy / footprint** | ⚠ minor: commit messages embed `Claude-Session:` URLs + `Co-Authored-By` — a traceable identifier in history (benign, removable if undesired) |

## Out of scope (flagged, not part of the delta)
The README's promotional/affiliate links (`bloome.im/login?ref=…`, `godmodeai.co`,
`crazyfaceai.com`) come from **upstream** (Yu Li) commits, not this fork's delta. If
a link-reputation pass is wanted, say so and I'll run it.

## Verdict
The fork's contribution is **clean and low-risk**: additive research + audit
material, **zero modification of the upstream library**, no secrets, no network or
dangerous code, stdlib-only. Maturity: documentation + self-tested lab prototypes
(not production, not a paradigm shift — see `FINAL_SCIENTIFIC_AUDIT.md`). Only minor
note: session identifiers in commit messages.
