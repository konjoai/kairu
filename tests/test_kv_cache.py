"""Tests for kairu.kv_cache — LRU logits cache + CachedModel wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from kairu.kv_cache import CachedModel, LogitsCache
from kairu.mock_model import MockModel
from kairu.speculative import SpeculativeDecoder


def test_cache_hit_miss_basic():
    c = LogitsCache(capacity=4)
    assert c.get((1, 2)) is None
    c.put((1, 2), np.array([1.0, 2.0]))
    out = c.get((1, 2))
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1.0, 2.0]))
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["size"] == 1


def test_cache_lru_eviction_order():
    c = LogitsCache(capacity=2)
    c.put((1,), np.zeros(2))
    c.put((2,), np.zeros(2))
    c.get((1,))  # marks (1,) as most recently used → (2,) is now LRU
    c.put((3,), np.zeros(2))  # evicts (2,)
    assert c.get((2,)) is None
    assert c.get((1,)) is not None
    assert c.get((3,)) is not None
    assert c.stats()["evictions"] == 1


def test_cache_update_existing_key_does_not_evict():
    c = LogitsCache(capacity=2)
    c.put((1,), np.zeros(2))
    c.put((2,), np.zeros(2))
    c.put((1,), np.ones(2))  # update, no evict
    assert c.stats()["evictions"] == 0
    np.testing.assert_array_equal(c.get((1,)), np.ones(2))


def test_cache_clear_resets_stats():
    c = LogitsCache(capacity=2)
    c.put((1,), np.zeros(2))
    c.get((1,))
    c.clear()
    s = c.stats()
    assert s["hits"] == 0 and s["misses"] == 0 and s["size"] == 0


def test_cache_rejects_bad_capacity():
    with pytest.raises(ValueError):
        LogitsCache(capacity=0)


def test_cache_hit_rate_math():
    c = LogitsCache(capacity=4)
    c.put((1,), np.zeros(2))
    c.get((1,))  # hit
    c.get((1,))  # hit
    c.get((2,))  # miss
    assert c.stats()["hit_rate"] == pytest.approx(2 / 3)


def test_cached_model_preserves_interface_contract():
    base = MockModel()
    m = CachedModel(base)
    assert m.vocab_size == base.vocab_size
    assert m.max_seq_len() == base.max_seq_len()
    out = m.next_token_logits([1, 2, 3])
    assert out.shape == (base.vocab_size,)
    assert out.dtype == np.float32


def test_cached_model_returns_identical_logits_on_hit():
    m = CachedModel(MockModel())
    a = m.next_token_logits([5, 5, 5])
    b = m.next_token_logits([5, 5, 5])
    np.testing.assert_array_equal(a, b)
    assert m.cache.stats()["hits"] == 1
    assert m.cache.stats()["misses"] == 1


def test_cached_model_helps_repeated_speculative_calls():
    """Two speculative.generate() calls on the same prompt must hit the cache."""
    target = CachedModel(MockModel(), cache_capacity=128)
    draft = MockModel()
    dec = SpeculativeDecoder(target, draft, gamma=4)
    dec.generate([1, 2, 3], max_new_tokens=8)
    misses_after_first = target.cache.stats()["misses"]
    dec.generate([1, 2, 3], max_new_tokens=8)
    s = target.cache.stats()
    # Second call must produce hits (the prompt-prefix logits are cached).
    assert s["hits"] > 0
    # And it must NOT add as many fresh misses as the first call.
    new_misses = s["misses"] - misses_after_first
    assert new_misses < misses_after_first
