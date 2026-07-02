"""
End-to-end CPU check: run the full AirLLM pipeline (split -> shards ->
layer-by-layer generation) on CPU, once with raw shards and once with
compression='lossless', and verify the outputs are token-for-token
identical -- the "lossless" guarantee checked on a real model through the
real pipeline.

Uses a tiny test model by default so it runs anywhere (needs network on
first run to download it). Pass any HF repo id to test a real model:

    python cpu_e2e_check.py                      # tiny test model
    python cpu_e2e_check.py Qwen/Qwen2.5-0.5B    # small real model
"""

import sys

sys.path.insert(0, '..')

from airllm import AutoModel

repo = sys.argv[1] if len(sys.argv) > 1 else 'hf-internal-testing/tiny-random-LlamaForCausalLM'
prompt = 'The theory of information was'

print(f"model: {repo}\n--- raw shards ---")
model = AutoModel.from_pretrained(repo, device='cpu', max_seq_len=64)
tokens = model.tokenizer([prompt], return_tensors='pt')
out_raw = model.generate(tokens['input_ids'], max_new_tokens=8, use_cache=True)
del model

print("--- lossless shards ---")
model = AutoModel.from_pretrained(repo, device='cpu', max_seq_len=64, compression='lossless')
out_lossless = model.generate(tokens['input_ids'], max_new_tokens=8, use_cache=True)

raw_ids = out_raw[0].tolist()
lossless_ids = out_lossless[0].tolist()
print(f"\nraw:      {raw_ids}")
print(f"lossless: {lossless_ids}")
print(f"token-for-token identical: {raw_ids == lossless_ids}")
if raw_ids != lossless_ids:
    raise SystemExit("ERROR: lossless path diverged from raw path!")
print(f"decoded: {model.tokenizer.decode(lossless_ids)!r}")
