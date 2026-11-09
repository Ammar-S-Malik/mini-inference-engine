"""Throughput vs batch size: our engine vs HuggingFace generate().

Usage:  python benchmark_batch.py            (fp32)
        python benchmark_batch.py --fp16
"""
import sys
import time

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from generate import generate_batched
from model import load_gpt2

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if "--fp16" in sys.argv else torch.float32
tag = "fp16" if dtype == torch.float16 else "fp32"
NAME = "MiniInfer"  # name of our engine in tables and charts

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
tok.padding_side = "left"
ours = load_gpt2(device, dtype)
hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype).eval()

base = [
    "The future of efficient AI inference is",
    "Once upon a time in a small village",
    "In 1969, the first humans to walk on the moon were",
    "def fibonacci(n):",
    "The capital of France is",
    "Machine learning models are trained by",
    "The best way to learn programming is",
    "Scientists recently discovered that",
]
N = 128
batch_sizes = [1, 2, 4, 8, 16, 32, 64]


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def make_batch(B):
    prompts = [base[i % len(base)] for i in range(B)]
    return tok(prompts, return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def run_ours(enc, n):
    return generate_batched(ours, enc.input_ids, enc.attention_mask, n)


@torch.no_grad()
def run_hf(enc, n):
    return hf.generate(
        enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=n,
        min_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id,
    )


def timed(fn, enc, repeats=3):
    fn(enc, 8)  # warmup
    sync()
    best = float("inf")
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn(enc, N)
        sync()
        best = min(best, time.perf_counter() - t0)
    return enc.input_ids.size(0) * N / best  # total tokens per second across the batch


rows = []
print(f"[{tag}] total generated tokens/sec ({N} new tokens per sequence)")
print(f"{'batch':>5} | {NAME:>9} | {'HF':>9} | {NAME} vs HF | {NAME} vs batch=1")
print("-" * 70)
base_ours = None
for B in batch_sizes:
    enc = make_batch(B)
    a, h = timed(run_ours, enc), timed(run_hf, enc)
    base_ours = base_ours or a
    rows.append((B, a, h))
    w1, w2 = len(f"{NAME} vs HF") - 1, len(f"{NAME} vs batch=1") - 1
    print(f"{B:>5} | {a:>9.0f} | {h:>9.0f} | {a / h:>{w1}.2f}x | {a / base_ours:>{w2}.1f}x")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r[0] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [r[1] for r in rows], "o-", label=f"{NAME} (static cache + SDPA, {tag})")
    plt.plot(xs, [r[2] for r in rows], "s--", label=f"HuggingFace generate ({tag})")
    plt.xscale("log", base=2)
    plt.xlabel("batch size (concurrent sequences)")
    plt.ylabel("total tokens / second")
    plt.title("GPT-2 small: throughput vs batch size")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = f"benchmark_batch_{tag}.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved chart to {out}")
except Exception as e:  # don't lose the table if plotting fails
    print(f"\n(chart skipped: {e})")
