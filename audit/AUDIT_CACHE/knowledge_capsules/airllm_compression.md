# Capsule: AirLLM weight compression

- **Hypothesis:** 4/8-bit compression shrinks layer weights with "tiny accuracy loss".
- **Evidence:** `compress_layer_state_dict` = bitsandbytes `quantize_nf4` /
  `quantize_blockwise`; inverse on load. `test_compression.py` asserts round-trip
  RMSE < 0.1 over 10 random matrices for both 4/8bit.
- **Counter-evidence:** test uses random Gaussian tensors, not real weights, and is
  GPU-only (`.cuda()`); "tiny accuracy loss" on actual model quality is **not**
  benchmarked here (no perplexity/task-accuracy test in-repo).
- **Risk:** end-to-end quality claim is asserted, not measured in this repo.
- **Confidence:** 95% the quant round-trip is correct; 45% the in-repo evidence
  supports the *quality* claim (it leans on bitsandbytes' external reputation).
- **Decision:** VALID but standard (wraps bitsandbytes). Quality claim = PROMISING,
  not PROVEN here. Add a perplexity benchmark to close the gap.
