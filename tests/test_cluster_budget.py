"""Tests for kairu.cluster_budget — LocalClusterBudget and ClusterTokenBudget.

All tests run fully offline. ClusterTokenBudget is tested via a mock Redis
client that replays the INCRBY / DECRBY / EXPIRE / GET / DELETE semantics.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from kairu.cluster_budget import (
    ClusterTokenBudget,
    LocalClusterBudget,
)


# ---------------------------------------------------------------------------
# Mock Redis client for ClusterTokenBudget unit tests
# ---------------------------------------------------------------------------

class _MockRedis:
    """Minimal in-process Redis mock supporting the commands kairu uses."""

    def __init__(self) -> None:
        self._store: dict[str, int] = {}

    async def incrby(self, key: str, n: int) -> int:
        self._store[key] = self._store.get(key, 0) + n
        return self._store[key]

    async def decrby(self, key: str, n: int) -> int:
        self._store[key] = self._store.get(key, 0) - n
        return self._store[key]

    async def expire(self, key: str, seconds: int) -> bool:
        return True  # no-op for the mock

    async def get(self, key: str):
        v = self._store.get(key)
        return str(v) if v is not None else None

    async def delete(self, key: str) -> int:
        return self._store.pop(key, None) and 1 or 0


# ---------------------------------------------------------------------------
# LocalClusterBudget tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_budget_allows_within_cap() -> None:
    budget = LocalClusterBudget(cap=100, window_s=60.0)
    granted = await budget.consume(50)
    assert granted == 50


@pytest.mark.asyncio
async def test_local_budget_rejects_over_cap() -> None:
    budget = LocalClusterBudget(cap=100, window_s=60.0)
    assert await budget.consume(80) == 80
    # 80 + 40 = 120 > 100 → must reject
    assert await budget.consume(40) == 0


@pytest.mark.asyncio
async def test_local_budget_remaining_decrements() -> None:
    budget = LocalClusterBudget(cap=200, window_s=60.0)
    await budget.consume(75)
    remaining = await budget.remaining()
    assert remaining == 125


@pytest.mark.asyncio
async def test_local_budget_utilization_fraction() -> None:
    budget = LocalClusterBudget(cap=100, window_s=60.0)
    await budget.consume(25)
    util = await budget.utilization()
    assert abs(util - 0.25) < 1e-9


@pytest.mark.asyncio
async def test_local_budget_reset_clears_counter() -> None:
    budget = LocalClusterBudget(cap=100, window_s=60.0)
    await budget.consume(90)
    await budget.reset()
    assert await budget.remaining() == 100


@pytest.mark.asyncio
async def test_local_budget_rejects_negative_consume() -> None:
    budget = LocalClusterBudget(cap=100, window_s=60.0)
    with pytest.raises(ValueError):
        await budget.consume(-1)


@pytest.mark.asyncio
async def test_local_budget_zero_consume_returns_zero() -> None:
    budget = LocalClusterBudget(cap=100, window_s=60.0)
    assert await budget.consume(0) == 0


@pytest.mark.asyncio
async def test_local_budget_exact_cap_boundary() -> None:
    """Consuming exactly cap tokens must succeed; one more must fail."""
    budget = LocalClusterBudget(cap=50, window_s=60.0)
    assert await budget.consume(50) == 50
    assert await budget.consume(1) == 0


@pytest.mark.asyncio
async def test_local_budget_invalid_cap() -> None:
    with pytest.raises(ValueError):
        LocalClusterBudget(cap=0, window_s=60.0)


@pytest.mark.asyncio
async def test_local_budget_invalid_window() -> None:
    with pytest.raises(ValueError):
        LocalClusterBudget(cap=100, window_s=0.0)


# ---------------------------------------------------------------------------
# ClusterTokenBudget (Redis-backed) unit tests via mock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_redis_budget_allows_within_cap() -> None:
    r = _MockRedis()
    budget = ClusterTokenBudget(r, cap=100, window_s=60.0)
    granted = await budget.consume(60)
    assert granted == 60


@pytest.mark.asyncio
async def test_redis_budget_rejects_over_cap() -> None:
    r = _MockRedis()
    budget = ClusterTokenBudget(r, cap=100, window_s=60.0)
    assert await budget.consume(80) == 80
    # 80 + 40 > 100 → rollback → 0
    assert await budget.consume(40) == 0
    # Counter must be back to 80 after rollback.
    remaining = await budget.remaining()
    assert remaining == 20


@pytest.mark.asyncio
async def test_redis_budget_remaining() -> None:
    r = _MockRedis()
    budget = ClusterTokenBudget(r, cap=200, window_s=60.0)
    await budget.consume(50)
    assert await budget.remaining() == 150


@pytest.mark.asyncio
async def test_redis_budget_utilization() -> None:
    r = _MockRedis()
    budget = ClusterTokenBudget(r, cap=100, window_s=60.0)
    await budget.consume(33)
    util = await budget.utilization()
    assert abs(util - 0.33) < 1e-9


@pytest.mark.asyncio
async def test_redis_budget_reset() -> None:
    r = _MockRedis()
    budget = ClusterTokenBudget(r, cap=100, window_s=60.0)
    await budget.consume(90)
    await budget.reset()
    # After reset the counter key is deleted; remaining = cap.
    assert await budget.remaining() == 100


@pytest.mark.asyncio
async def test_redis_budget_invalid_cap() -> None:
    r = _MockRedis()
    with pytest.raises(ValueError):
        ClusterTokenBudget(r, cap=0, window_s=60.0)


@pytest.mark.asyncio
async def test_redis_budget_invalid_window() -> None:
    r = _MockRedis()
    with pytest.raises(ValueError):
        ClusterTokenBudget(r, cap=100, window_s=-1.0)


@pytest.mark.asyncio
async def test_redis_budget_scope_isolates_keys() -> None:
    """Different scope strings must use independent counters."""
    r = _MockRedis()
    b1 = ClusterTokenBudget(r, cap=100, window_s=60.0, scope="tenant-A")
    b2 = ClusterTokenBudget(r, cap=100, window_s=60.0, scope="tenant-B")
    await b1.consume(90)
    # b2 has not consumed anything — still full.
    assert await b2.remaining() == 100


@pytest.mark.asyncio
async def test_redis_budget_zero_consume() -> None:
    r = _MockRedis()
    budget = ClusterTokenBudget(r, cap=100, window_s=60.0)
    assert await budget.consume(0) == 0
