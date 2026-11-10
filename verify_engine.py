"""Continuous batching must produce the same tokens as generating each request alone.

Uses fewer slots than requests, so slots are freed and refilled mid-run.
"""
import torch
from transformers import AutoTokenizer

from engine import Engine, Request
from generate import generate_cached
from model import load_gpt2

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
model = load_gpt2(device)

prompts = [
    "The future of efficient AI inference is",
    "Once upon a time",
    "In 1969, the first humans to walk on the moon were",
    "def fibonacci(n):",
    "The capital of France is",
    "Machine learning models are trained by",
    "The best way to learn programming is",
    "Scientists recently discovered that",
]
new_tokens = [12, 40, 25, 8, 33, 18, 45, 22]  # different lengths -> slots finish at different times

reqs = [Request(i, tok(p).input_ids, n) for i, (p, n) in enumerate(zip(prompts, new_tokens))]
stats = Engine(model, max_slots=3, max_len=128).run(reqs)  # 8 requests through 3 slots

ok = True
for r, p in zip(reqs, prompts):
    ids = torch.tensor([r.prompt_ids], device=device)
    ref = generate_cached(model, ids, r.max_new_tokens)[0, len(r.prompt_ids):].tolist()
    same = ref == r.tokens
    ok &= same
    print(f"[{'OK ' if same else 'BAD'}] req {r.rid}: {r.max_new_tokens:>2} tokens  {p!r}")
    if not same:
        print("   alone :", tok.decode(ref))
        print("   engine:", tok.decode(r.tokens))

print(f"\n{stats['tokens']} tokens in {stats['steps']} decode steps (3 slots, 8 requests)")
assert ok, "continuous batching output differs from one-at-a-time output"
print("All continuous-batching outputs match single-sequence outputs.")
