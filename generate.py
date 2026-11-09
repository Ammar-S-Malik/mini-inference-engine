"""Greedy generation loops: naive (recompute everything) vs. KV-cached."""
import torch

from model import KVCache


@torch.no_grad()
def generate_naive(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Every step re-runs the model over the ENTIRE sequence so far."""
    ids = input_ids
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


@torch.no_grad()
def generate_cached(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Prefill once over the prompt, then feed ONE new token per step."""
    cache = KVCache(model.cfg.n_layer)
    logits = model(input_ids, cache)[:, -1, :]  # prefill: fills the cache
    ids = input_ids
    for step in range(max_new_tokens):
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if step < max_new_tokens - 1:
            logits = model(nxt, cache)[:, -1, :]  # decode: only the new token
    return ids


@torch.no_grad()
def generate_batched(model, input_ids, attention_mask, max_new_tokens):
    """Greedy decoding for a LEFT-padded batch, using the preallocated cache.

    input_ids / attention_mask: (B, P). Every sequence gets exactly
    max_new_tokens new tokens (EOS is ignored, so benchmarks are comparable).
    """
    from model import StaticKVCache

    B, P = input_ids.shape
    cfg = model.cfg
    p = next(model.parameters())
    cache = StaticKVCache(
        cfg.n_layer, B, cfg.n_head, P + max_new_tokens, cfg.n_embd // cfg.n_head, p.device, p.dtype
    )
    mask = attention_mask
    logits = model(input_ids, cache, attention_mask=mask)[:, -1, :]  # prefill
    out = []
    for step in range(max_new_tokens):
        nxt = logits.argmax(dim=-1, keepdim=True)
        out.append(nxt)
        if step < max_new_tokens - 1:
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=mask.device)], dim=1)
            logits = model(nxt, cache, attention_mask=mask)[:, -1, :]  # decode
    return torch.cat([input_ids] + out, dim=1)
