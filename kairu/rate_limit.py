"""Pluggable per-key rate limiter — in-memory and Redis backends.

Both backends implement the same sliding-window contract: at time ``t``,
allow a request from ``key`` iff

    |{u ∈ history(key) : t - u ≤ W}| < N

In-memory uses a ``deque`` per key; Redis uses a sorted set with
``ZADD`` / ``ZREMRANGEBYSCORE`` / ``ZCARD`` inside a ``MULTI`` so the
operation is atomic across racing app servers — the single-source-of-truth
property the in-memory limiter loses the moment you horizontally scale.

The :class:`RateLimiter` glue layer is what the server holds; it picks
the backend at construction time. Public API is unchanged from v0.4.0
(``await limiter.check(key)``) — existing tests still pass.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Protocol


class RateLimiterBackend(Protocol):
    """Storage contract for sliding-window rate limit state.

    All methods are async so Redis (or any future remote store) can plug in
    without changing the call sites.
    """

    async def allow(self, key: str, now: float, max_requests: int, window_s: float) -> bool:
        """Return True iff the request should be permitted; record it on True."""
        ...


@dataclass
class _Bucket:
    times: deque = field(default_factory=deque)


class InMemoryBackend:
    """Single-process sliding window. O(1) amortized per check.

    Math: maintain a deque of timestamps per key. On ``allow()``:

      1. evict every stamp ≤ ``now - window_s``
      2. if remaining count ≥ ``max_requests`` → reject
      3. else append ``now`` and accept

    The ``asyncio.Lock`` makes it correct under concurrent task interleaving
    inside the same event loop.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str, now: float, max_requests: int, window_s: float) -> bool:
        async with self._lock:
            b = self._buckets.setdefault(key, _Bucket())
            cutoff = now - window_s
            while b.times and b.times[0] <= cutoff:
                b.times.popleft()
            if len(b.times) >= max_requests:
                return False
            b.times.append(now)
            return True

    def __len__(self) -> int:
        return len(self._buckets)


class RedisBackend:
    """Sliding window backed by a Redis sorted set.

    Each key gets a ZSET ``kairu:rl:{key}`` whose members are stamp strings
    and whose scores are the same float timestamps. The atomic pipeline:

        ZREMRANGEBYSCORE k 0 (now-W)   # evict expired
        ZCARD            k             # count remaining
        ZADD             k now now     # speculatively add this request
        EXPIRE           k W*2         # bound memory if a key goes idle

    If ``ZCARD`` (the count *before* this add) ≥ N, we issue a follow-up
    ``ZREM`` to undo the speculative add. The window then reflects the true
    count without leaking. Two round-trips in the worst case, one in the
    common case — the trade-off vs. an EVAL/Lua script is portability and
    debuggability; Lua is a Phase 7 optimization if a real workload demands it.
    """

    KEY_PREFIX = "kairu:rl:"

    def __init__(self, redis_client) -> None:  # redis.asyncio.Redis
        self._r = redis_client

    async def allow(self, key: str, now: float, max_requests: int, window_s: float) -> bool:
        full_key = f"{self.KEY_PREFIX}{key}"
        cutoff = now - window_s
        member = f"{now:.9f}"

        pipe = self._r.pipeline(transaction=True)
        pipe.zremrangebyscore(full_key, 0, cutoff)
        pipe.zcard(full_key)
        pipe.zadd(full_key, {member: now})
        pipe.expire(full_key, max(1, int(window_s * 2)))
        results = await pipe.execute()
        # results = [evicted_count, prior_zcard, zadd_added, expire_set]
        prior_count = int(results[1])
        if prior_count >= max_requests:
            await self._r.zrem(full_key, member)
            return False
        return True


class RateLimiter:
    """Public glue. Constructed by :class:`kairu.server.create_app`.

    Backwards-compatible with the v0.4.0 in-memory limiter — passing nothing
    keeps the same behavior. Pass a ``RedisBackend`` instance to share state
    across processes.
    """

    def __init__(
        self,
        max_requests: int,
        window_s: float,
        backend: Optional[RateLimiterBackend] = None,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._max = max_requests
        self._win = window_s
        # NB: explicit None check — InMemoryBackend defines __len__,
        # so an empty backend evaluates falsy under `or`.
        self._backend: RateLimiterBackend = backend if backend is not None else InMemoryBackend()

    @property
    def backend(self) -> RateLimiterBackend:
        return self._backend

    async def check(self, key: str, now: Optional[float] = None) -> bool:
        t = time.monotonic() if now is None else now
        return await self._backend.allow(key, t, self._max, self._win)


__all__ = [
    "RateLimiter",
    "RateLimiterBackend",
    "InMemoryBackend",
    "RedisBackend",
]
