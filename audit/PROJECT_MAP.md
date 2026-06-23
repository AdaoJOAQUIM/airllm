# PROJECT_MAP (Phase 0/1)

Repo: `adaojoaquim/airllm` fork. ~8,113 LoC Python. Three independent zones.

## Module graph (core)

```
AutoModel (auto_model.py)            # detect model type -> pick adapter
      |
AirLLM<Model> adapters               # airllm_llama/mistral/qwen/.../mixtral (thin)
      |
AirLLMBaseModel (airllm_base.py 642) # orchestrates LAYER-BY-LAYER forward
      |          \
utils.py (403)    persist/ (IO)       # split/compress/load layers; safetensors/mlx
      |
bitsandbytes (4/8bit) + safetensors + huggingface_hub
```

Concept → Module → Claim:
- **Layered streaming inference** → `utils.split_and_save_layers`, `load_layer`,
  `airllm_base` forward loop → "run 70B on 4GB / 405B on 8GB VRAM".
- **Weight compression** → `utils.compress/uncompress_layer_state_dict` → 4/8-bit
  quantization via bitsandbytes.

## Criticality classification

| Zone | Files | Class | Why |
|---|---|---|---|
| airllm core | `airllm_base.py`, `utils.py`, `persist/*`, `auto_model.py` | **CRITICAL** | the actual product; all claims live here |
| model adapters | `airllm_{llama,mistral,qwen,qwen2,mixtral,chatglm,internlm,baichuan}*` | Important | thin per-arch wrappers |
| research/tme | `cee.py`, `tme.py`, `orchestrator.py`, `igc.py`, `abstraction.py`, `engine.py`, `adaptive_compute_probe.py` | Important (separate) | this session's prototypes; self-tested |
| training/finetune | `training/qlora.py`, `rlhf/qlora_dpo.py`, `anima_100k/*`, `scripts/*`, `eval/*` | **Secondary (pruned)** | standard QLoRA/DPO/long-context; no novel claim |

## Entry points
- Library: `from airllm import AutoModel` → `AutoModel.from_pretrained(...)`.
- Research: each `research/**/*.py` has `--selftest` / `--demo`.

## Test surface (real)
- `air_llm/tests/test_compression.py` — 4/8bit round-trip RMSE<0.1 (**GPU-only**, `.cuda()`).
- `air_llm/tests/test_automodel.py` — adapter selection.
- `research/**` — deterministic `--selftest` in every module (CPU, no GPU/LLM).
- Gap: no CPU/integration test of the layered-inference path itself.
