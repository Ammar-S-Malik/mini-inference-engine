"""GPT-2 (124M) written from scratch, with an optional KV cache.

Everything in here is plain PyTorch ops on purpose: the point of the project is
that you understand (and can later optimize) every line of the forward pass.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


class KVCache:
    """Stores the keys/values of every layer so past tokens are never recomputed.

    Each tensor has shape (batch, n_head, seq_len, head_dim) and grows along
    dim=2 every time new tokens are appended.
    """

    def __init__(self, n_layer: int):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].size(2)

    def advance(self, n: int):  # grows by itself in update(); nothing to do
        pass


class StaticKVCache:
    """Preallocated cache: one big buffer per layer, written in place.

    No torch.cat and no reallocation per step. Buffers have shape
    (batch, n_head, max_len, head_dim); `length` says how much is filled.
    """

    def __init__(self, n_layer, batch, n_head, max_len, head_dim, device, dtype):
        shape = (batch, n_head, max_len, head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.length = 0
        self.max_len = max_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        T, s = k.size(2), self.length
        assert s + T <= self.max_len, "static cache is full"
        self.k[layer][:, :, s : s + T] = k
        self.v[layer][:, :, s : s + T] = v
        return self.k[layer][:, :, : s + T], self.v[layer][:, :, : s + T]

    @property
    def seq_len(self) -> int:
        return self.length

    def advance(self, n: int):  # called once per forward, after all layers wrote
        self.length += n


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, layer_idx: int, cache=None, mask=None):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, head_dim)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)  # k, v now cover past + new tokens

        # mask: bool, True = this query may attend to this key. Shape (B|1, 1, T, S).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # fused kernel
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.mlp = MLP(cfg)

    def forward(self, x, layer_idx: int, cache=None, mask=None):
        x = x + self.attn(self.ln_1(x), layer_idx, cache, mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config = GPT2Config()):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)  # position embeddings
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)

    def forward(self, idx, cache=None, attention_mask=None, position_ids=None):
        """idx: (B, T) token ids -> logits (B, T, vocab).

        attention_mask: optional (B, past+T) tensor of 1 (real token) / 0 (padding),
        covering cached tokens AND the new ones. Used for left-padded batches.
        """
        B, T = idx.shape
        past = cache.seq_len if cache is not None else 0
        S = past + T
        assert S <= self.cfg.n_ctx, "sequence longer than context window"
        dev = idx.device

        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(past, S, device=dev)
            else:  # position = number of real tokens before this one
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, past:]

        # Causal mask (T, S): query i sits at absolute position S-T+i.
        mask = torch.ones(T, S, dtype=torch.bool, device=dev).tril(diagonal=S - T)[None, None]
        if attention_mask is not None:
            mask = mask & attention_mask.bool()[:, None, None, :]
            # Let every position see itself so a padding row is never fully masked
            # (an all-masked row would produce NaNs).
            diag = torch.zeros(T, S, dtype=torch.bool, device=dev)
            diag[torch.arange(T, device=dev), torch.arange(T, device=dev) + (S - T)] = True
            mask = mask | diag[None, None]

        x = self.wte(idx) + self.wpe(position_ids)
        for i, block in enumerate(self.h):
            x = block(x, i, cache, mask)
        if cache is not None:
            cache.advance(T)
        x = self.ln_f(x)
        return F.linear(x, self.wte.weight)  # lm_head is tied to wte


def load_gpt2(device: str = "cuda", dtype: torch.dtype = torch.float32) -> GPT2:
    """Load the pretrained GPT-2 small weights from HuggingFace into our model."""
    from transformers import GPT2LMHeadModel

    sd = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()

    def g(key: str) -> torch.Tensor:  # key naming differs slightly across versions
        return sd[key] if key in sd else sd["transformer." + key]

    cfg = GPT2Config()
    new = {"wte.weight": g("wte.weight"), "wpe.weight": g("wpe.weight")}
    for i in range(cfg.n_layer):
        p = f"h.{i}."
        for ln in ("ln_1", "ln_2"):
            new[p + ln + ".weight"] = g(p + ln + ".weight")
            new[p + ln + ".bias"] = g(p + ln + ".bias")
        for name in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            # HF stores these as Conv1D (in, out); nn.Linear wants (out, in).
            new[p + name + ".weight"] = g(p + name + ".weight").t().contiguous()
            new[p + name + ".bias"] = g(p + name + ".bias")
    new["ln_f.weight"] = g("ln_f.weight")
    new["ln_f.bias"] = g("ln_f.bias")

    model = GPT2(cfg)
    model.load_state_dict(new)  # strict: any naming mistake fails loudly
    return model.to(device=device, dtype=dtype).eval()
