"""Tests for kairu.kv_cache — LRU logits cache + CachedModel wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from kairu.kv_cache import CachedModel, LogitsCache, QuantizedArray, _quantize
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


# --------------------------------------------------------------------------- #
# Quantised storage tier (int8 / int4)
# --------------------------------------------------------------------------- #


def test_quantize_int8_roundtrip_within_one_step():
    arr = np.linspace(-3.0, 5.0, 100).astype(np.float32)
    qa = _quantize(arr, 8)
    deq = qa.dequantize()
    assert deq.shape == arr.shape
    assert deq.dtype == np.float32
    # Affine min-max rounding error is bounded by half a quantisation step.
    assert np.max(np.abs(deq - arr)) <= qa.scale / 2 + 1e-5


def test_quantize_int4_roundtrip_coarser_but_bounded():
    arr = np.linspace(-3.0, 5.0, 100).astype(np.float32)
    qa = _quantize(arr, 4)
    deq = qa.dequantize()
    assert np.max(np.abs(deq - arr)) <= qa.scale / 2 + 1e-5
    # 15 levels span the range → much coarser than int8's 255.
    assert qa.scale > _quantize(arr, 8).scale


def test_quantize_int4_packs_two_codes_per_byte():
    arr = np.zeros(100, dtype=np.float32)
    arr[:] = np.arange(100) % 16  # exercise all nibble values
    qa = _quantize(arr, 4)
    assert qa.bits == 4
    assert qa.n == 100
    assert qa.data.dtype == np.uint8
    assert qa.nbytes() == 50  # ceil(100 / 2)


def test_quantize_int4_odd_length_padded_and_recovered():
    arr = np.linspace(0.0, 1.0, 7).astype(np.float32)
    qa = _quantize(arr, 4)
    assert qa.nbytes() == 4  # ceil(7 / 2)
    assert qa.dequantize().shape == (7,)


def test_quantize_constant_array_is_exact():
    arr = np.full(10, 2.5, dtype=np.float32)
    deq = _quantize(arr, 8).dequantize()
    np.testing.assert_array_almost_equal(deq, arr)


def test_cache_quant_int8_shrinks_footprint_and_returns_approx():
    c = LogitsCache(capacity=4, quant="int8")
    arr = np.linspace(-2.0, 2.0, 100).astype(np.float32)
    c.put((1,), arr)
    out = c.get((1,))
    assert out is not None
    # int8 codes: 100 bytes vs 400 for raw float32.
    assert c.stats()["memory_bytes"] == 100
    np.testing.assert_array_equal(out, _quantize(arr, 8).dequantize())


def test_cache_quant_none_stores_raw_bitexact():
    c = LogitsCache(capacity=4, quant="none")
    arr = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    c.put((1,), arr)
    np.testing.assert_array_equal(c.get((1,)), arr)
    assert c.stats()["memory_bytes"] == arr.nbytes


def test_cache_rejects_bad_quant_mode():
    with pytest.raises(ValueError):
        LogitsCache(quant="int3")


def test_quantized_array_is_frozen():
    qa = _quantize(np.arange(4, dtype=np.float32), 8)
    assert isinstance(qa, QuantizedArray)
    with pytest.raises(Exception):
        qa.scale = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Attention-weighted (heavy-hitter) eviction
# --------------------------------------------------------------------------- #


def test_cache_rejects_bad_eviction_policy():
    with pytest.raises(ValueError):
        LogitsCache(eviction="random")


def test_attention_eviction_drops_lowest_attention_entry():
    c = LogitsCache(capacity=2, eviction="attention")
    c.put((1,), np.zeros(2))
    c.put((2,), np.zeros(2))
    c.get((1,))  # (1,) attention → 2.0, (2,) stays at 1.0
    c.put((3,), np.zeros(2))  # evicts the lighter entry (2,)
    assert c.get((2,)) is None
    assert c.get((1,)) is not None
    assert c.get((3,)) is not None
    assert c.stats()["evictions"] == 1


def test_attention_eviction_protects_heavy_hitter_against_recency():
    """A heavy hitter survives even when it is the least-recently-used entry —
    the behaviour that distinguishes the attention policy from plain LRU."""
    c = LogitsCache(capacity=2, eviction="attention")
    c.put((1,), np.zeros(2))
    c.put((2,), np.zeros(2))
    c.add_attention((1,), 5.0)  # (1,) is now the heavy hitter (weight 6.0)
    c.get((2,))  # (2,) becomes most-recently-used; LRU would protect it
    c.put((3,), np.zeros(2))
    assert c.get((1,)) is not None  # heavy hitter kept despite being LRU
    assert c.get((2,)) is None


def test_attention_eviction_ties_break_on_oldest():
    c = LogitsCache(capacity=2, eviction="attention")
    c.put((1,), np.zeros(2))
    c.put((2,), np.zeros(2))
    c.put((3,), np.zeros(2))  # equal weights → evict the oldest, (1,)
    assert c.get((1,)) is None
    assert c.get((2,)) is not None


def test_add_attention_reports_presence():
    c = LogitsCache(capacity=2, eviction="attention")
    c.put((1,), np.zeros(2))
    assert c.add_attention((1,), 3.0) is True
    assert c.add_attention((99,), 3.0) is False  # absent key → no-op


def test_stats_exposes_policy_and_quant_defaults():
    s = LogitsCache().stats()
    assert s["eviction"] == "lru"
    assert s["quant"] == "none"
    assert s["memory_bytes"] == 0


def test_clear_resets_attention_weights():
    c = LogitsCache(capacity=2, eviction="attention")
    c.put((1,), np.zeros(2))
    c.add_attention((1,), 9.0)
    c.clear()
    # After clear the key is gone, so add_attention must report absence.
    assert c.add_attention((1,), 1.0) is False
    assert len(c) == 0


# --------------------------------------------------------------------------- #
# CachedModel forwarding of the new knobs
# --------------------------------------------------------------------------- #


def test_cached_model_quant_returns_dequantized_logits_on_hit():
    m = CachedModel(MockModel(), quant="int8")
    a = m.next_token_logits([5, 5, 5])  # miss → exact computed logits
    b = m.next_token_logits([5, 5, 5])  # hit → dequantised approximation
    np.testing.assert_array_equal(b, _quantize(a, 8).dequantize())
    assert m.cache.stats()["hits"] == 1
    assert m.cache.stats()["quant"] == "int8"


def test_cached_model_attention_policy_forwarded():
    m = CachedModel(MockModel(), cache_capacity=8, eviction="attention")
    assert m.cache.stats()["eviction"] == "attention"
    out = m.next_token_logits([1, 2, 3])
    assert out.shape == (m.vocab_size,)
