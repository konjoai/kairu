"""Logits cache + cached model wrapper — recycle target-model calls across
speculative verification steps.

Why this matters
----------------
In speculative decoding the target model is called on overlapping prefixes::

    verify step 1:  target.next_token_logits([p, t1])
    verify step 2:  target.next_token_logits([p, t1, t2])
    bonus token:    target.next_token_logits([p, t1, t2, t3])
    next iteration: target.next_token_logits([p, t1, t2, t3])  # ← redundant

A real KV cache stores per-layer K/V tensors keyed by sequence position. We
expose the same *behavior* at the model-interface level by memoizing the final
logits keyed by ``tuple(token_ids)``. This is sound because logits are a pure
function of the prefix for any deterministic model (and our :class:`MockModel`
is fully deterministic by construction).

The cache is bounded LRU (``OrderedDict.move_to_end``) — O(1) get, O(1) put.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from kairu.base import ModelInterface


class LogitsCache:
    """Bounded LRU cache mapping ``tuple[int, ...] -> np.ndarray``.

    Keys are token-id tuples (hashable, immutable). Values are stored without
    copying — callers must not mutate returned arrays. Stats are exposed via
    :meth:`stats` for observability.
    """

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._store: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: tuple[int, ...]) -> np.ndarray | None:
        v = self._store.get(key)
        if v is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return v

    def put(self, key: tuple[int, ...], value: np.ndarray) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if len(self._store) >= self._cap:
            self._store.popitem(last=False)
            self._evictions += 1
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._store),
            "capacity": self._cap,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
        }


class CachedModel(ModelInterface):
    """Wrap any :class:`ModelInterface` with logits memoization.

    Drop-in replacement: same shape/dtype contract as the underlying model.
    Use ``model.cache.stats()`` to observe hit rate after a generation.
    """

    def __init__(self, base: ModelInterface, cache_capacity: int = 128) -> None:
        self._base = base
        self.cache = LogitsCache(capacity=cache_capacity)

    @property
    def vocab_size(self) -> int:
        return self._base.vocab_size

    def max_seq_len(self) -> int:
        return self._base.max_seq_len()

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        key = tuple(token_ids)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        out = self._base.next_token_logits(token_ids)
        self.cache.put(key, out)
        return out


__all__ = ["LogitsCache", "CachedModel"]
