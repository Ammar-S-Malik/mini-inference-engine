"""Batched generation (left padding + static cache) must match one-at-a-time generation."""
import torch
from transformers import AutoTokenizer

from generate import generate_batched, generate_cached
from model import load_gpt2

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = load_gpt2(device)

prompts = [
    "The future of efficient AI inference is",
    "Once upon a time",
    "In 1969, the first humans to walk on the moon were",
    "def fibonacci(n):",
    "The capital of France is",
]
N = 30

enc = tok(prompts, return_tensors="pt", padding=True).to(device)
batched = generate_batched(model, enc.input_ids, enc.attention_mask, N)
P = enc.input_ids.size(1)

ok = True
for i, prompt in enumerate(prompts):
    single_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    single = generate_cached(model, single_ids, N)[0, single_ids.size(1):]
    got = batched[i, P:]
    same = torch.equal(single, got)
    ok &= same
    print(f"[{'OK ' if same else 'BAD'}] {prompt!r} (prompt len {single_ids.size(1)}, padded to {P})")
    if not same:
        print("   single :", tok.decode(single))
        print("   batched:", tok.decode(got))

assert ok, "batched output differs from one-at-a-time output"
print("\nAll batched outputs match single-sequence outputs.")
