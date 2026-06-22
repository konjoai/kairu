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

Two storage/eviction strategies are available, both opt-in so the default
path stays bit-exact and identical to prior releases:

* **Eviction** — ``"lru"`` (recency, the default) or ``"attention"``, an
  H2O-style *heavy-hitter* policy (Zhang et al. 2023, NeurIPS) that evicts the
  entry with the least accumulated attention mass rather than the oldest one.
  Cache hits accrue attention automatically; :meth:`LogitsCache.add_attention`
  injects external scores (e.g. a real model's attention rollup).
* **Quantisation** — ``"none"`` (the default), ``"int8"`` or ``"int4"``.
  Cached arrays are stored under affine min-max quantisation, shrinking the
  cache's resident footprint at the cost of a small, bounded precision loss on
  retrieval. ``int4`` values are packed two-per-byte for a true 4-bit tier.

The cache is bounded — O(1) get/put under LRU; eviction under the attention
policy is O(n) in the cache size (one ``min`` scan), paid only on insert.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from kairu.base import ModelInterface

_EVICTION_POLICIES = ("lru", "attention")
_QUANT_MODES = ("none", "int8", "int4")
# Each access (a cache hit) contributes this much attention mass to the entry.
_HIT_ATTENTION = 1.0


def _pack_int4(codes: np.ndarray) -> np.ndarray:
    """Pack a uint8 array of 4-bit codes (0..15) two-per-byte, high nibble first."""
    flat = codes.astype(np.uint8)
    if flat.size % 2 == 1:
        flat = np.append(flat, np.uint8(0))
    pairs = flat.reshape(-1, 2)
    return ((pairs[:, 0] << 4) | pairs[:, 1]).astype(np.uint8)


def _unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_pack_int4`; recover the first ``n`` 4-bit codes."""
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = (packed >> 4) & 0x0F
    out[1::2] = packed & 0x0F
    return out[:n]


@dataclass(frozen=True)
class QuantizedArray:
    """An affine min-max quantised 1-D float array.

    ``x ≈ code * scale + offset``. Codes are unsigned integers in ``[0, 2^bits
    - 1]``; for ``bits == 4`` they are packed two-per-byte in :attr:`data`.
    """

    data: np.ndarray
    scale: float
    offset: float
    n: int
    bits: int

    def nbytes(self) -> int:
        """Resident size of the stored codes in bytes (excludes scalar metadata)."""
        return int(self.data.nbytes)

    def dequantize(self) -> np.ndarray:
        """Reconstruct the approximate ``float32`` array from the stored codes."""
        codes = _unpack_int4(self.data, self.n) if self.bits == 4 else self.data
        return (
            codes.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)
        ).astype(np.float32)


def _quantize(arr: np.ndarray, bits: int) -> QuantizedArray:
    """Affine min-max quantise ``arr`` to ``bits`` (8 or 4) unsigned levels."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    lo = float(flat.min())
    hi = float(flat.max())
    levels = (1 << bits) - 1
    span = hi - lo
    # A constant array has zero span: every value maps to code 0 and is
    # reconstructed exactly as the offset, so scale is irrelevant — use 1.0.
    scale = span / levels if span > 0.0 else 1.0
    codes = np.clip(np.round((flat - lo) / scale), 0, levels).astype(np.uint8)
    data = _pack_int4(codes) if bits == 4 else codes
    return QuantizedArray(data=data, scale=scale, offset=lo, n=flat.size, bits=bits)


class LogitsCache:
    """Bounded cache mapping ``tuple[int, ...] -> np.ndarray``.

    Keys are token-id tuples (hashable, immutable). Under the default
    ``eviction="lru"`` / ``quant="none"`` configuration values are stored
    without copying and returned bit-exact — callers must not mutate returned
    arrays. Enabling quantisation makes retrieval lossy but shrinks the
    footprint; enabling the attention policy switches eviction from recency to
    accumulated attention mass. Stats are exposed via :meth:`stats`.
    """

    def __init__(
        self,
        capacity: int = 128,
        *,
        eviction: str = "lru",
        quant: str = "none",
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if eviction not in _EVICTION_POLICIES:
            raise ValueError(
                f"eviction must be one of {_EVICTION_POLICIES}, got {eviction!r}"
            )
        if quant not in _QUANT_MODES:
            raise ValueError(f"quant must be one of {_QUANT_MODES}, got {quant!r}")
        self._cap = capacity
        self._eviction = eviction
        self._quant = quant
        self._store: OrderedDict[tuple[int, ...], np.ndarray | QuantizedArray] = (
            OrderedDict()
        )
        self._weight: dict[tuple[int, ...], float] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: tuple[int, ...]) -> np.ndarray | None:
        """Return the (dequantised) value for ``key`` or ``None`` on a miss."""
        stored = self._store.get(key)
        if stored is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._weight[key] = self._weight.get(key, 0.0) + _HIT_ATTENTION
        self._hits += 1
        return stored.dequantize() if isinstance(stored, QuantizedArray) else stored

    def put(self, key: tuple[int, ...], value: np.ndarray) -> None:
        """Insert or update ``key``; evict per policy when at capacity."""
        stored: np.ndarray | QuantizedArray = (
            _quantize(value, 8 if self._quant == "int8" else 4)
            if self._quant != "none"
            else value
        )
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = stored
            return
        if len(self._store) >= self._cap:
            self._evict_one()
        self._store[key] = stored
        self._weight[key] = _HIT_ATTENTION

    def add_attention(self, key: tuple[int, ...], weight: float) -> bool:
        """Add external attention mass to a live entry's heavy-hitter score.

        Returns ``True`` if ``key`` is resident (and was updated), ``False`` if
        it is absent — adding attention to an already-evicted key is expected,
        not an error, so it is a no-op rather than a raise.
        """
        if key not in self._weight:
            return False
        self._weight[key] += weight
        return True

    def _evict_one(self) -> None:
        """Remove exactly one entry according to the active eviction policy."""
        if self._eviction == "attention":
            # Lowest accumulated attention; ``min`` breaks ties on the first
            # (oldest, least-recently-used) key in insertion/recency order.
            victim = min(self._store, key=lambda k: self._weight.get(k, 0.0))
            del self._store[victim]
        else:
            victim, _ = self._store.popitem(last=False)
        self._weight.pop(victim, None)
        self._evictions += 1

    def memory_bytes(self) -> int:
        """Approximate resident bytes of all stored values."""
        return int(
            sum(
                v.nbytes() if isinstance(v, QuantizedArray) else v.nbytes
                for v in self._store.values()
            )
        )

    def clear(self) -> None:
        """Drop all entries and reset hit/miss/eviction counters."""
        self._store.clear()
        self._weight.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        """Return a snapshot of cache counters, config, and resident size."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._store),
            "capacity": self._cap,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
            "eviction": self._eviction,
            "quant": self._quant,
            "memory_bytes": self.memory_bytes(),
        }


class CachedModel(ModelInterface):
    """Wrap any :class:`ModelInterface` with logits memoization.

    Drop-in replacement: same shape/dtype contract as the underlying model.
    Use ``model.cache.stats()`` to observe hit rate after a generation. The
    ``eviction`` and ``quant`` knobs forward to the backing :class:`LogitsCache`
    — note that ``quant != "none"`` makes cached logits approximate.
    """

    def __init__(
        self,
        base: ModelInterface,
        cache_capacity: int = 128,
        *,
        eviction: str = "lru",
        quant: str = "none",
    ) -> None:
        self._base = base
        self.cache = LogitsCache(
            capacity=cache_capacity, eviction=eviction, quant=quant
        )

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


__all__ = ["LogitsCache", "CachedModel", "QuantizedArray"]
