"""Continuous batching engine.

A fixed pool of `max_slots` sequence slots. Every step decodes ALL active slots
in one forward pass. As soon as a request finishes, its slot is freed and the
next waiting request is prefilled into it, so the batch never idles waiting for
its longest member (which is what static batching does).
"""
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from model import SlotKVCache, StaticKVCache


@dataclass
class Request:
    rid: int
    prompt_ids: list
    max_new_tokens: int
    arrival: float = 0.0  # seconds after the run starts
    tokens: list = field(default_factory=list)  # generated tokens
    t_first: float = None  # time the first token was produced
    t_done: float = None


class Engine:
    def __init__(self, model, max_slots: int, max_len: int):
        self.model = model
        self.S = max_slots
        self.max_len = max_len
        cfg = model.cfg
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.cache = SlotKVCache(
            cfg.n_layer, max_slots, cfg.n_head, max_len, cfg.n_embd // cfg.n_head, self.device, self.dtype
        )

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _prefill(self, req: Request, slot: int):
        """Run the prompt through the model alone, then copy its KV into the slot."""
        cfg = self.model.cfg
        P = len(req.prompt_ids)
        ids = torch.tensor([req.prompt_ids], device=self.device)
        tmp = StaticKVCache(cfg.n_layer, 1, cfg.n_head, P, cfg.n_embd // cfg.n_head, self.device, self.dtype)
        logits = self.model(ids, tmp)[:, -1, :]
        for l in range(cfg.n_layer):
            self.cache.k[l][slot, :, :P] = tmp.k[l][0, :, :P]
            self.cache.v[l][slot, :, :P] = tmp.v[l][0, :, :P]
        self.cache.lens[slot] = P
        return int(logits.argmax(dim=-1))  # first generated token

    @torch.no_grad()
    def run(self, requests):
        """Serve all requests; returns timing stats. Requests are mutated in place."""
        for r in requests:
            assert len(r.prompt_ids) + r.max_new_tokens <= self.max_len, "request too long for max_len"
        S = self.S
        waiting = deque(sorted(requests, key=lambda r: r.arrival))
        slots = [None] * S  # slot -> Request
        lens_py = [0] * S  # python mirror of cache.lens (avoids GPU syncs)
        act = torch.zeros(S, dtype=torch.long, device=self.device)
        cur = torch.zeros(S, 1, dtype=torch.long, device=self.device)  # last token per slot
        self.cache.lens.zero_()
        steps = 0

        self._sync()
        t0 = time.perf_counter()
        while waiting or any(slots):
            now = time.perf_counter() - t0
            # 1) admit waiting requests into free slots
            for s in range(S):
                if slots[s] is None and waiting and waiting[0].arrival <= now:
                    r = waiting.popleft()
                    first = self._prefill(r, s)
                    r.tokens = [first]
                    r.t_first = time.perf_counter() - t0
                    slots[s], lens_py[s] = r, len(r.prompt_ids)
                    act[s] = 1
                    cur[s, 0] = first
                    if len(r.tokens) >= r.max_new_tokens:  # 1-token request
                        r.t_done = r.t_first
                        slots[s], lens_py[s] = None, 0
                        act[s] = 0
                        self.cache.lens[s] = 0
            if not any(slots):
                if waiting:
                    time.sleep(0.0005)  # nothing running, next request not here yet
                continue

            # 2) one decode step for all active slots
            kv_len = max(lens_py[s] for s in range(S) if slots[s] is not None) + 1
            logits = self.model.decode_slots(cur, self.cache, kv_len)
            nxt = logits.argmax(dim=-1)
            self.cache.lens += act
            cur = nxt[:, None].clone()
            out = nxt.tolist()  # one GPU->CPU sync per step
            steps += 1
            t = time.perf_counter() - t0

            # 3) record tokens, free finished slots
            for s in range(S):
                r = slots[s]
                if r is None:
                    continue
                lens_py[s] += 1
                r.tokens.append(out[s])
                if len(r.tokens) >= r.max_new_tokens:
                    r.t_done = t
                    slots[s], lens_py[s] = None, 0
                    act[s] = 0
                    self.cache.lens[s] = 0

        self._sync()
        total = time.perf_counter() - t0
        n_tok = sum(len(r.tokens) for r in requests)
        return {"seconds": total, "tokens": n_tok, "tok_per_s": n_tok / total, "steps": steps}
