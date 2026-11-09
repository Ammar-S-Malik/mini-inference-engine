# MiniInfer

A small LLM inference engine, written from scratch in PyTorch. It runs GPT-2 (124M) with a hand-written forward pass, a KV cache, a preallocated static cache, and left-padded batched generation, and is benchmarked against HuggingFace's `generate()`.

The goal is to understand, and measure, the optimizations that production servers such as vLLM are built on.

## Results

Hardware: NVIDIA RTX 5060, PyTorch (CUDA 12.8), greedy decoding, 128 new tokens per sequence. Figures are exact benchmark output (best of 3 runs).

**Batched throughput (total tokens/sec across all sequences)**

| | batch 1 | batch 64 | vs. HuggingFace at batch 64 |
|---|---|---|---|
| MiniInfer fp32 | 189 | 8,508 | 1.42x |
| MiniInfer fp16 | 193 | 12,245 | 1.54x |

Batching raises total throughput by 45x (fp32) and 63x (fp16) over one-request-at-a-time generation. HuggingFace's `generate()` batches too, so the fair comparison is the right-hand column. The lead over HuggingFace ranges from 1.41x to 1.74x in fp32 and from 1.30x to 1.65x in fp16 across batch sizes 1 to 64. It is largest at small and medium batches and narrows at batch 64.

![fp16 throughput vs batch size](benchmark_batch_fp16.png)
![fp32 throughput vs batch size](benchmark_batch_fp32.png)

**KV cache, single sequence.** Without a cache, throughput falls as the sequence grows (each step recomputes everything). With the cache it stays flat, giving about 2.3x at 512 new tokens, and about 1.3x over HuggingFace's own cached loop.

![naive vs KV cache vs HuggingFace](benchmark_vs_hf.png)

At batch size 1 the engine is bound by Python and kernel-launch overhead (~5 ms per token), not by GPU compute. That is why batching helps so much, and why fp16 does little for a single request.

## Correctness

Everything is checked, not assumed:

- `verify.py`: logits match HuggingFace GPT-2 (max difference ~1e-4), greedy output matches HuggingFace token for token, and the KV-cached loop equals the naive loop.
- `verify_batch.py`: prompts of different lengths, left-padded into one batch, produce the same tokens as generating each prompt alone.

## What's implemented

- GPT-2 forward pass in plain PyTorch, loading the pretrained HuggingFace weights
- Dynamic KV cache, then a preallocated static KV cache (no per-step reallocation)
- Fused attention via `scaled_dot_product_attention` with explicit causal and padding masks
- Left-padded batched generation with per-sequence position ids
- fp32 and fp16 benchmarks against HuggingFace

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers matplotlib

python verify.py            # model matches HuggingFace
python verify_batch.py      # batched == one-at-a-time
python benchmark.py         # KV cache benchmark
python benchmark_batch.py   # throughput vs batch size (add --fp16 for half precision)
```

RTX 50-series (Blackwell) GPUs need the CUDA 12.8 build of PyTorch. Python 3.12 or newer is required (the benchmark scripts use f-string syntax from 3.12).

## Files

| File | Purpose |
|---|---|
| `model.py` | GPT-2 architecture, KV caches, HuggingFace weight loader |
| `generate.py` | Naive, KV-cached and batched greedy decoding |
| `verify.py` / `verify_batch.py` | Correctness checks |
| `benchmark.py` / `benchmark_batch.py` | Benchmarks and charts |

## Roadmap

- [x] From-scratch forward pass, verified against HuggingFace
- [x] KV cache and static cache
- [x] Fused attention and fp16
- [x] Batched generation with padding
- [ ] Continuous batching scheduler
- [ ] Paged KV cache (block allocator)
- [ ] Speculative decoding
- [ ] FastAPI streaming server with a load test
- [ ] CUDA graphs and a custom Triton attention kernel

## Limitations

Greedy decoding only (no sampling yet), GPT-2 small only, one GPU, and timings come from a desktop PC, so run-to-run noise of a few percent is normal.
