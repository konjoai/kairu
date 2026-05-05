"""Cluster-scoped token budget enforced via shared Redis counters.

Motivation
----------
The per-request :class:`kairu.budget.TokenBudget` enforces a local cap on a
single generation.  At cluster scale (multiple kairu replicas behind a load
balancer) you also need a *global* ceiling: e.g. "no more than 5 M tokens/day
across all pods."  This module provides that via Redis atomic ``INCRBY`` +
``EXPIRE`` counters.

Design
------
* A :class:`ClusterTokenBudget` owns a Redis key named
  ``kairu:budget:<scope>`` where *scope* defaults to ``"global"`` but can be
  any string (``"tenant:<id>"``, ``"model:gpt2"``, etc.).
* The counter resets automatically when its TTL expires — use ``window_s`` to
  model hourly / daily budgets.
* ``consume(n)`` is atomic: ``INCRBY`` returns the *new* total; if that
  exceeds the cap the call rolls back via ``DECRBY`` and returns 0.
* The :class:`LocalClusterBudget` stub implements the same interface purely
  in-process for tests that run without Redis.

All I/O is async (``redis.asyncio`` or any compatible client).

Usage::

    import redis.asyncio as aioredis
    from kairu.cluster_budget import ClusterTokenBudget

    redis = aioredis.from_url("redis://localhost:6379")
    budget = ClusterTokenBudget(redis, cap=1_000_000, window_s=86400)

    tokens_granted = await budget.consume(256)  # 256 or 0 (budget exhausted)
    remaining      = await budget.remaining()
    usage_fraction = await budget.utilization()
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Protocol


# ---------------------------------------------------------------------------
# Shared interface
# ---------------------------------------------------------------------------

class ClusterBudgetBackend(Protocol):
    """Storage contract for cluster-scoped token budgets."""

    async def consume(self, n: int) -> int:
        """Attempt to consume *n* tokens.  Returns *n* on success, 0 if the
        budget is exhausted."""
        ...

    async def remaining(self) -> int:
        """Return the number of tokens remaining in the current window."""
        ...

    async def utilization(self) -> float:
        """Return usage as a fraction in ``[0.0, 1.0]``."""
        ...

    async def reset(self) -> None:
        """Reset the counter to 0 (useful in tests and admin tooling)."""
        ...


# ---------------------------------------------------------------------------
# Redis-backed implementation
# ---------------------------------------------------------------------------

class ClusterTokenBudget:
    """Cluster-scoped token budget backed by a Redis counter.

    Args:
        redis_client: An ``redis.asyncio.Redis`` instance (or compatible mock).
        cap:          Maximum tokens allowed per *window_s* seconds.
        window_s:     Sliding window length in seconds.  The Redis key's TTL
                      is set to ``ceil(window_s)`` on first write so the
                      counter resets automatically.
        scope:        Key prefix distinguishing different budgets on the same
                      Redis instance.  Defaults to ``"global"``.
    """

    def __init__(
        self,
        redis_client,
        cap: int,
        window_s: float = 3600.0,
        scope: str = "global",
    ) -> None:
        if cap < 1:
            raise ValueError("cap must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._r = redis_client
        self._cap = cap
        self._window_s = window_s
        self._key = f"kairu:budget:{scope}"
        self._window_int = max(1, int(window_s))

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def window_s(self) -> float:
        return self._window_s

    async def consume(self, n: int) -> int:
        """Atomically consume *n* tokens.

        Returns *n* when the budget has room; returns 0 and rolls back when
        the cap would be exceeded.

        Math: ``INCRBY`` returns the new total T.
          * T ≤ cap → accepted, set TTL on first write (``EXPIRE`` only if the
            key didn't exist yet, avoiding resetting a live window).
          * T > cap → ``DECRBY n`` to roll back; return 0.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return 0

        new_total: int = await self._r.incrby(self._key, n)
        if new_total <= self._cap:
            # Set expiry only when the counter was just created (new_total == n)
            # to avoid resetting a window that is already ticking.
            if new_total == n:
                await self._r.expire(self._key, self._window_int)
            return n
        else:
            await self._r.decrby(self._key, n)
            return 0

    async def remaining(self) -> int:
        """Tokens left in the current window (always >= 0)."""
        raw = await self._r.get(self._key)
        used = int(raw) if raw is not None else 0
        return max(0, self._cap - used)

    async def utilization(self) -> float:
        """Usage as a fraction ``[0.0, 1.0]``."""
        raw = await self._r.get(self._key)
        used = int(raw) if raw is not None else 0
        return min(1.0, used / self._cap)

    async def reset(self) -> None:
        """Delete the counter key (resets to 0 for the next window)."""
        await self._r.delete(self._key)


# ---------------------------------------------------------------------------
# In-process stub — for tests and single-process deployments
# ---------------------------------------------------------------------------

class LocalClusterBudget:
    """In-process token budget — same interface, no Redis required.

    Thread-safe via ``asyncio.Lock``.  Suitable for:
      * Unit tests that run without Redis.
      * Single-process deployments that want cluster-budget semantics
        without external infrastructure.

    Args:
        cap:      Maximum tokens per window.
        window_s: Window length in seconds.  The counter resets after this
                  many seconds from the first ``consume()`` call.
    """

    def __init__(self, cap: int, window_s: float = 3600.0) -> None:
        if cap < 1:
            raise ValueError("cap must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._cap = cap
        self._window_s = window_s
        self._used: int = 0
        self._window_start: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def window_s(self) -> float:
        return self._window_s

    async def consume(self, n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return 0

        async with self._lock:
            now = time.monotonic()
            # Reset if the window has expired.
            if self._window_start is not None and (now - self._window_start) >= self._window_s:
                self._used = 0
                self._window_start = None

            if self._window_start is None:
                self._window_start = now

            if self._used + n > self._cap:
                return 0
            self._used += n
            return n

    async def remaining(self) -> int:
        async with self._lock:
            self._maybe_reset()
            return max(0, self._cap - self._used)

    async def utilization(self) -> float:
        async with self._lock:
            self._maybe_reset()
            return min(1.0, self._used / self._cap)

    async def reset(self) -> None:
        async with self._lock:
            self._used = 0
            self._window_start = None

    def _maybe_reset(self) -> None:
        """Called inside the lock; resets if the window has expired."""
        if self._window_start is not None:
            now = time.monotonic()
            if (now - self._window_start) >= self._window_s:
                self._used = 0
                self._window_start = None


__all__ = [
    "ClusterBudgetBackend",
    "ClusterTokenBudget",
    "LocalClusterBudget",
]
