# Capsule: AirLLM layered-streaming inference

- **Hypothesis:** A 70B/405B model runs on 4-8GB VRAM by loading one layer at a time.
- **Evidence:** `utils.split_and_save_layers` shards a checkpoint into per-layer
  safetensors; `load_layer`/`uncompress_layer_state_dict` load+dequant one layer;
  `airllm_base` runs forward layer-by-layer, freeing memory between layers
  (`clean_memory`). Mechanism is real and standard; library is on PyPI, widely used.
- **Counter-evidence:** none for *memory*; the saving is genuine.
- **Risk:** latency. Weights are re-read from disk **every token**, so throughput is
  very low vs a resident model — a memory↔speed tradeoff, not free lunch.
- **Confidence:** 92% it works as claimed for memory; 90% it is I/O/latency-bound.
- **Decision:** VALID engineering. NOT a new paradigm — a classic space-time tradeoff
  (offload + standard quantization). Maturity: production.
