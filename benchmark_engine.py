"""Continuous batching vs static batching on a mixed-length workload.

Workload: R requests with different output lengths, all waiting at time 0.
Static batching runs them in fixed groups; every group lasts as long as its
LONGEST request, so short requests leave their slot idle. Continuous batching
refills a slot the moment its request finishes.

Throughput counts only USEFUL tokens (what each request actually asked for).

Usage:  python benchmark_engine.py            (fp32)
        python benchmark_engine.py --fp16
"""
import random
import sys
import time

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from engine import Engine, Request
from generate import generate_batched
from model import load_gpt2

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if "--fp16" in sys.argv else torch.float32
tag = "fp16" if dtype == torch.float16 else "fp32"
NAME = "MiniInfer"

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = load_gpt2(device, dtype)
hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype).eval()

prompts = [
    "The future of efficient AI inference is",
    "Once upon a time in a small village",
    "In 1969, the first humans to walk on the moon were",
    "def fibonacci(n):",
    "The capital of France is",
    "Machine learning models are trained by",
    "The best way to learn programming is",
    "Scientists recently discovered that",
]
R = 64
rng = random.Random(0)
lengths = [rng.choice([16, 24, 32, 48, 64, 96, 128, 192, 256]) for _ in range(R)]  # skewed mix
work = [(prompts[i % len(prompts)], lengths[i]) for i in range(R)]
useful = sum(lengths)
MAX_LEN = 64 + 256  # prompt (< 64 tokens) + longest output


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def make_requests():
    return [Request(i, tok(p).input_ids, n) for i, (p, n) in enumerate(work)]


def run_continuous(slots):
    eng = Engine(model, slots, MAX_LEN)
    stats = eng.run(make_requests())
    return stats["seconds"]


@torch.no_grad()
def run_static(slots, use_hf=False):
    sync()
    t0 = time.perf_counter()
    for i in range(0, R, slots):
        chunk = work[i : i + slots]
        n = max(c[1] for c in chunk)  # the whole group runs until its longest request is done
        enc = tok([c[0] for c in chunk], return_tensors="pt", padding=True).to(device)
        if use_hf:
            hf.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=n,
                        min_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        else:
            generate_batched(model, enc.input_ids, enc.attention_mask, n)
    sync()
    return time.perf_counter() - t0


def best(fn, *args, repeats=2):
    fn(*args)  # warmup
    return min(fn(*args) for _ in range(repeats))


print(f"[{tag}] {R} requests, output lengths 16-256 (mean {useful / R:.0f}), {useful} useful tokens")
print(f"{'slots':>5} | {'HF static':>10} | {NAME + ' static':>16} | {NAME + ' continuous':>20} | vs static | vs HF")
print("-" * 82)
rows = []
for slots in [4, 8, 16, 32]:
    t_hf = best(run_static, slots, True)
    t_st = best(run_static, slots, False)
    t_ct = best(run_continuous, slots)
    rows.append((slots, useful / t_hf, useful / t_st, useful / t_ct))
    print(f"{slots:>5} | {useful / t_hf:>10.0f} | {useful / t_st:>16.0f} | {useful / t_ct:>20.0f} | {t_st / t_ct:>8.2f}x | {t_hf / t_ct:>4.2f}x")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r[0] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [r[3] for r in rows], "o-", label=f"{NAME} continuous batching")
    plt.plot(xs, [r[2] for r in rows], "o--", label=f"{NAME} static batching")
    plt.plot(xs, [r[1] for r in rows], "s--", label="HuggingFace generate (static)")
    plt.xscale("log", base=2)
    plt.xlabel("slots (max concurrent sequences)")
    plt.ylabel("useful tokens / second")
    plt.title(f"Mixed-length workload, {R} requests ({tag})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = f"benchmark_engine_{tag}.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved chart to {out}")
except Exception as e:
    print(f"\n(chart skipped: {e})")
