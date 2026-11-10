# Design

MiniInfer is a small LLM inference engine for GPT-2 (124M). This page explains how the pieces fit together.

## The model (`model.py`)

The forward pass is written in plain PyTorch ops (embeddings, layer norm, attention, MLP) and loads the pretrained HuggingFace weights, so its output can be compared to the reference directly. Attention uses `scaled_dot_product_attention` with an explicit boolean mask. The mask is built once per forward pass and combines causal masking with padding.

## Three KV caches

Generating a token needs the keys and values of every earlier token. Recomputing them each step is the main cost of naive generation, so they are cached.

| Cache | How it works | Used by |
|---|---|---|
| `KVCache` | Grows with `torch.cat` each step | Single-sequence generation |
| `StaticKVCache` | One preallocated buffer per layer, written in place, one shared length | Batched generation |
| `SlotKVCache` | Preallocated buffers with an independent length per slot | Continuous batching |

## Batching with padding

For a batch of prompts with different lengths, prompts are left-padded to one length. An attention mask marks padding, and position ids count only real tokens, so a padded prompt behaves exactly like the same prompt alone. Each padding position is allowed to attend to itself so no row of the mask is fully masked (an all-masked row produces NaNs).

## Continuous batching (`engine.py`)

The engine owns a fixed pool of slots. Each loop iteration does three things:

1. **Admit.** If a slot is free and a request is waiting, run its prompt through the model alone, copy the resulting keys and values into the slot, and record its first token.
2. **Decode.** Run one forward pass for all slots at once (`GPT2.decode_slots`). Each slot writes its new key and value at its own position and attends over its own prefix, using a mask built from the per-slot lengths.
3. **Retire.** Record the new tokens. When a request reaches its length, free its slot so the next iteration can refill it.

Static batching waits for the longest request in a group. Continuous batching never lets a finished slot sit idle, which is where its throughput gain comes from. Cache buffers are zero-initialised because masked positions must hold finite values (0 times NaN is NaN).

## How correctness is checked

Every optimization has a test that compares its output with a simpler reference: the model against HuggingFace, the cached loop against the naive loop, batched generation against one-at-a-time generation, and continuous batching against one-at-a-time generation. Benchmarks are only run after the matching test passes.

## Known limitations and next steps

- Prefill runs one request at a time and pauses decoding. Production engines chunk or overlap it.
- Every slot's buffer is sized for the longest possible request. A paged cache would allocate memory in small blocks instead.
- A single request is limited by kernel-launch overhead, not GPU compute. CUDA graphs or `torch.compile` would attack that.
- Speculative decoding (draft with a small model, verify with GPT-2) would cut single-request latency.
