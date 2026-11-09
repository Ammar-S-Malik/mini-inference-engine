"""Benchmark naive generation vs. KV-cached generation."""
import time

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from generate import generate_cached, generate_naive
from model import load_gpt2

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
model = load_gpt2(device)
ids = tok("The future of efficient AI inference is", return_tensors="pt").input_ids.to(device)
hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()


@torch.no_grad()
def generate_hf(_model, input_ids, max_new_tokens):
    """HuggingFace's own generate(), forced to produce exactly N tokens."""
    return hf.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, n_tokens: int, repeats: int = 3) -> float:
    fn(model, ids, 8)  # warmup
    sync()
    best = float("inf")
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn(model, ids, n_tokens)
        sync()
        best = min(best, time.perf_counter() - t0)
    return n_tokens / best  # tokens per second


NAME = "MiniInfer"  # name of our engine in tables and charts
lengths = [32, 64, 128, 256, 512]
rows = []
print(f"{'new tokens':>10} | {'naive tok/s':>12} | {NAME + '+cache':>12} | {'HF generate':>12} | {NAME} vs naive | {NAME} vs HF")
print("-" * 82)
for n in lengths:
    a, b, c = timed(generate_naive, n), timed(generate_cached, n), timed(generate_hf, n)
    rows.append((n, a, b, c))
    print(f"{n:>10} | {a:>12.1f} | {b:>12.1f} | {c:>12.1f} | {b / a:>{len(f"{NAME} vs naive") - 1}.2f}x | {b / c:>{len(f"{NAME} vs HF") - 1}.2f}x")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r[0] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [r[1] for r in rows], "o-", label="naive")
    plt.plot(xs, [r[2] for r in rows], "o-", label=f"{NAME} + KV cache")
    plt.plot(xs, [r[3] for r in rows], "s--", label="HuggingFace generate")
    plt.xlabel("new tokens generated")
    plt.ylabel("tokens / second")
    plt.title(f"GPT-2 small: naive vs {NAME} KV cache vs HuggingFace")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_vs_hf.png", dpi=150)
    print("\nSaved chart to benchmark_vs_hf.png")
except ImportError:
    print("\n(matplotlib not installed - skipping chart)")
