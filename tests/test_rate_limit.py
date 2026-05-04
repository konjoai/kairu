"""Tests for kairu.rate_limit — pluggable backends + RedisBackend logic."""
from __future__ import annotations

import pytest

from kairu.rate_limit import (
    InMemoryBackend,
    RateLimiter,
    RedisBackend,
)


# ─── public RateLimiter API (unchanged from v0.4.0) ───────────────────────

@pytest.mark.asyncio
async def test_in_memory_default_backend():
    rl = RateLimiter(max_requests=2, window_s=10.0)
    assert isinstance(rl.backend, InMemoryBackend)


@pytest.mark.asyncio
async def test_in_memory_sliding_window():
    rl = RateLimiter(max_requests=2, window_s=10.0)
    assert await rl.check("a", now=0.0) is True
    assert await rl.check("a", now=1.0) is True
    assert await rl.check("a", now=2.0) is False
    assert await rl.check("a", now=11.0) is True  # first stamp expired
    assert await rl.check("b", now=2.0) is True   # different key


@pytest.mark.asyncio
async def test_in_memory_independent_keys():
    rl = RateLimiter(max_requests=1, window_s=60.0)
    assert await rl.check("client-a", now=0.0) is True
    assert await rl.check("client-b", now=0.0) is True
    assert await rl.check("client-a", now=0.0) is False


@pytest.mark.asyncio
async def test_rate_limiter_rejects_bad_config():
    with pytest.raises(ValueError):
        RateLimiter(max_requests=0, window_s=1.0)
    with pytest.raises(ValueError):
        RateLimiter(max_requests=1, window_s=0.0)


# ─── RedisBackend logic — unit-tested with a mock redis client ────────────

class _MockPipe:
    """Just enough redis pipeline behavior for the backend's use case."""

    def __init__(self, store: dict[str, list[tuple[float, str]]]):
        self._store = store
        self._cmds: list = []

    def zremrangebyscore(self, key, mn, mx):
        self._cmds.append(("evict", key, mn, mx))
        return self

    def zcard(self, key):
        self._cmds.append(("zcard", key))
        return self

    def zadd(self, key, mapping):
        self._cmds.append(("zadd", key, mapping))
        return self

    def expire(self, key, seconds):
        self._cmds.append(("expire", key, seconds))
        return self

    async def execute(self):
        results = []
        for cmd in self._cmds:
            op = cmd[0]
            key = cmd[1]
            entries = self._store.setdefault(key, [])
            if op == "evict":
                _, _, mn, mx = cmd
                kept = [(s, m) for s, m in entries if not (mn <= s <= mx)]
                results.append(len(entries) - len(kept))
                self._store[key] = kept
            elif op == "zcard":
                results.append(len(entries))
            elif op == "zadd":
                _, _, mapping = cmd
                added = 0
                for member, score in mapping.items():
                    if not any(m == member for _, m in entries):
                        entries.append((score, member))
                        added += 1
                results.append(added)
            elif op == "expire":
                results.append(True)
        return results


class _MockRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[tuple[float, str]]] = {}

    def pipeline(self, transaction: bool = True):
        return _MockPipe(self.store)

    async def zrem(self, key: str, member: str):
        entries = self.store.get(key, [])
        self.store[key] = [(s, m) for s, m in entries if m != member]


@pytest.mark.asyncio
async def test_redis_backend_allow_then_reject():
    r = _MockRedis()
    backend = RedisBackend(r)
    rl = RateLimiter(max_requests=2, window_s=10.0, backend=backend)
    assert await rl.check("a", now=0.0) is True
    assert await rl.check("a", now=1.0) is True
    assert await rl.check("a", now=2.0) is False
    # Speculative add must have been rolled back — store has 2, not 3.
    assert len(r.store["kairu:rl:a"]) == 2


@pytest.mark.asyncio
async def test_redis_backend_eviction():
    r = _MockRedis()
    backend = RedisBackend(r)
    rl = RateLimiter(max_requests=2, window_s=10.0, backend=backend)
    await rl.check("a", now=0.0)
    await rl.check("a", now=1.0)
    # 11s later → both expired → allowed again
    assert await rl.check("a", now=11.0) is True


@pytest.mark.asyncio
async def test_redis_backend_independent_keys():
    r = _MockRedis()
    backend = RedisBackend(r)
    rl = RateLimiter(max_requests=1, window_s=60.0, backend=backend)
    assert await rl.check("alpha", now=0.0) is True
    assert await rl.check("beta", now=0.0) is True
    assert await rl.check("alpha", now=0.0) is False


@pytest.mark.asyncio
async def test_backend_can_be_swapped_through_constructor():
    backend = InMemoryBackend()
    rl = RateLimiter(max_requests=1, window_s=10.0, backend=backend)
    assert rl.backend is backend
    assert await rl.check("x", now=0.0) is True
    assert await rl.check("x", now=0.5) is False


@pytest.mark.asyncio
async def test_in_memory_backend_tracks_buckets():
    backend = InMemoryBackend()
    rl = RateLimiter(max_requests=2, window_s=5.0, backend=backend)
    await rl.check("u1", now=0.0)
    await rl.check("u2", now=0.0)
    assert len(backend) == 2
