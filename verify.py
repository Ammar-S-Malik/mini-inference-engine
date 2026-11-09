"""Correctness check: our from-scratch GPT-2 must match HuggingFace's."""
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from generate import generate_cached, generate_naive
from model import load_gpt2

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
prompt = "The future of efficient AI inference is"
ids = tok(prompt, return_tensors="pt").input_ids.to(device)

ours = load_gpt2(device)
hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

with torch.no_grad():
    diff = (ours(ids) - hf(ids).logits).abs().max().item()
print(f"1) max |logit difference| vs HuggingFace: {diff:.2e}")
assert diff < 1e-3, "forward pass does not match HuggingFace"

N = 40
with torch.no_grad():
    ref = hf.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=N,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
naive = generate_naive(ours, ids, N)
cached = generate_cached(ours, ids, N)

print(f"2) naive  == HuggingFace greedy: {torch.equal(naive, ref)}")
print(f"3) cached == naive:              {torch.equal(cached, naive)}")
assert torch.equal(cached, naive), "KV cache changed the output - there is a bug"

print("\nSample output:\n" + tok.decode(cached[0]))
print("\nAll checks passed.")
