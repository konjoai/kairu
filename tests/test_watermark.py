"""Tests for kairu.watermark — WatermarkLogitsProcessor and WatermarkDetector.

All tests are fully offline (NumPy + stdlib only).  No HF / torch required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kairu.mock_model import MockModel
from kairu.streaming import StreamingDecoder
from kairu.watermark import (
    WatermarkDetector,
    WatermarkLogitsProcessor,
    WatermarkResult,
    _make_green_list,
    _norm_sf,
    _seed_from_context,
    _seed_from_token,
)

VOCAB = 1000


# ---------------------------------------------------------------------------
# _seed helpers
# ---------------------------------------------------------------------------

def test_seed_from_token_deterministic():
    """Same token always yields the same seed."""
    s1 = _seed_from_token(42)
    s2 = _seed_from_token(42)
    assert s1 == s2


def test_seed_from_token_distinct():
    """Different tokens yield different seeds with overwhelming probability."""
    seeds = {_seed_from_token(i) for i in range(200)}
    assert len(seeds) == 200


def test_seed_from_context_deterministic():
    ctx = [10, 20, 30]
    assert _seed_from_context(ctx, window=3) == _seed_from_context(ctx, window=3)


def test_seed_from_context_window_trims():
    """Context window trims the prefix; only the tail matters."""
    tail = [99, 100]
    long_ctx = [1, 2, 3] + tail
    short_ctx = tail
    # window=2 ⇒ only last 2 tokens hashed
    assert _seed_from_context(long_ctx, window=2) == _seed_from_context(short_ctx, window=2)


# ---------------------------------------------------------------------------
# _make_green_list
# ---------------------------------------------------------------------------

def test_green_list_shape_and_dtype():
    mask = _make_green_list(VOCAB, seed=42, green_fraction=0.5)
    assert mask.shape == (VOCAB,)
    assert mask.dtype == bool


def test_green_list_fraction():
    """Half the vocab should be green (±1 due to floor)."""
    mask = _make_green_list(VOCAB, seed=7, green_fraction=0.5)
    n_green = int(mask.sum())
    assert abs(n_green - VOCAB // 2) <= 1


def test_green_list_reproducible():
    m1 = _make_green_list(VOCAB, seed=123, green_fraction=0.5)
    m2 = _make_green_list(VOCAB, seed=123, green_fraction=0.5)
    np.testing.assert_array_equal(m1, m2)


def test_green_list_changes_with_seed():
    m1 = _make_green_list(VOCAB, seed=0, green_fraction=0.5)
    m2 = _make_green_list(VOCAB, seed=1, green_fraction=0.5)
    # Two random permutations should differ almost everywhere
    assert not np.array_equal(m1, m2)


# ---------------------------------------------------------------------------
# WatermarkLogitsProcessor construction
# ---------------------------------------------------------------------------

def test_processor_bad_vocab():
    with pytest.raises(ValueError, match="vocab_size"):
        WatermarkLogitsProcessor(vocab_size=1)


def test_processor_bad_delta():
    with pytest.raises(ValueError, match="delta"):
        WatermarkLogitsProcessor(vocab_size=VOCAB, delta=-1.0)


def test_processor_bad_scheme():
    with pytest.raises(ValueError, match="seeding_scheme"):
        WatermarkLogitsProcessor(vocab_size=VOCAB, seeding_scheme="bogus")


# ---------------------------------------------------------------------------
# WatermarkLogitsProcessor.process
# ---------------------------------------------------------------------------

def test_process_green_tokens_biased_up():
    """Green tokens should have logits strictly higher than baseline."""
    proc = WatermarkLogitsProcessor(vocab_size=VOCAB, delta=2.0)
    base = np.zeros(VOCAB, dtype=np.float32)
    context = [5, 10, 15]
    biased = proc.process(base, context_ids=context)
    green = proc.green_list(context)
    # All green tokens got +delta; all red tokens unchanged
    np.testing.assert_allclose(biased[green], 2.0)
    np.testing.assert_allclose(biased[~green], 0.0)


def test_process_does_not_mutate_input():
    """process() must return a copy, never modify the input array."""
    proc = WatermarkLogitsProcessor(vocab_size=VOCAB, delta=3.0)
    logits = np.ones(VOCAB, dtype=np.float32)
    original = logits.copy()
    proc.process(logits, context_ids=[1, 2])
    np.testing.assert_array_equal(logits, original)


def test_process_shape_mismatch_raises():
    proc = WatermarkLogitsProcessor(vocab_size=VOCAB)
    with pytest.raises(ValueError, match="vocab_size"):
        proc.process(np.zeros(VOCAB + 1), context_ids=[1])


def test_process_empty_context_uses_fixed_seed():
    """Empty context must not crash and must be deterministic."""
    proc = WatermarkLogitsProcessor(vocab_size=VOCAB, delta=1.5)
    logits = np.zeros(VOCAB, dtype=np.float32)
    out1 = proc.process(logits, context_ids=[])
    out2 = proc.process(logits, context_ids=[])
    np.testing.assert_array_equal(out1, out2)


def test_context_scheme_differs_from_single():
    """Context seeding scheme should produce a different green list than single."""
    ctx = [10, 20, 30]
    p_single = WatermarkLogitsProcessor(vocab_size=VOCAB, seeding_scheme="single")
    p_context = WatermarkLogitsProcessor(
        vocab_size=VOCAB, seeding_scheme="context", context_window=3
    )
    g_single = p_single.green_list(ctx)
    g_context = p_context.green_list(ctx)
    # They use different seeds → masks should differ
    assert not np.array_equal(g_single, g_context)


# ---------------------------------------------------------------------------
# WatermarkDetector construction
# ---------------------------------------------------------------------------

def test_detector_bad_threshold():
    with pytest.raises(ValueError, match="z_threshold"):
        WatermarkDetector(vocab_size=VOCAB, z_threshold=0.0)


# ---------------------------------------------------------------------------
# WatermarkDetector.detect
# ---------------------------------------------------------------------------

def test_detect_empty_raises():
    det = WatermarkDetector(vocab_size=VOCAB)
    with pytest.raises(ValueError, match="non-empty"):
        det.detect(token_ids=[])


def test_detect_result_fields():
    det = WatermarkDetector(vocab_size=VOCAB)
    result = det.detect(token_ids=list(range(50)), prefix_ids=[1, 2, 3])
    assert isinstance(result, WatermarkResult)
    assert result.num_tokens == 50
    assert result.num_green + (50 - result.num_green) == 50
    assert 0.0 <= result.green_fraction <= 1.0
    assert isinstance(result.z_score, float)
    assert 0.0 <= result.p_value <= 1.0
    assert isinstance(result.decision, bool)


def test_detect_watermarked_sequence_high_z():
    """A sequence generated with the watermark processor should yield a high z-score.

    We generate a long watermarked sequence (500 tokens) and verify the z-score
    exceeds 2.0 — a directional check (not a hard threshold) to keep the test
    deterministic regardless of MockModel internals.
    """
    proc = WatermarkLogitsProcessor(vocab_size=VOCAB, delta=5.0, green_fraction=0.5)
    model = MockModel()
    decoder = StreamingDecoder(model, temperature=1.0, watermark=proc)
    prompt = [1, 2, 3]
    generated = decoder.generate(prompt, max_new_tokens=500)

    det = WatermarkDetector(vocab_size=VOCAB, green_fraction=0.5, z_threshold=4.0)
    result = det.detect(token_ids=generated, prefix_ids=prompt)
    assert result.z_score > 2.0, (
        f"Expected z > 2.0 for watermarked text, got {result.z_score:.3f}"
    )
    assert result.green_fraction > 0.5, (
        f"Expected >50 %% green tokens, got {result.green_fraction:.3f}"
    )


def test_detect_unwatermarked_sequence_low_z():
    """A plain (unwatermarked) sequence should have a z-score near zero on average.

    We run without the processor and verify the green fraction is plausibly near
    0.5 — allowing ±3 standard deviations (extremely conservative).
    """
    model = MockModel()
    decoder = StreamingDecoder(model, temperature=1.0)  # no watermark
    prompt = [1, 2, 3]
    generated = decoder.generate(prompt, max_new_tokens=400)

    det = WatermarkDetector(vocab_size=VOCAB, green_fraction=0.5, z_threshold=4.0)
    result = det.detect(token_ids=generated, prefix_ids=prompt)
    # Under H0 (no watermark) we expect |z| < 4 with P > 0.9999
    # MockModel is deterministic, so this will always pass for fixed seeds.
    assert not result.decision, (
        f"Unwatermarked text should not trigger detection, z={result.z_score:.3f}"
    )


# ---------------------------------------------------------------------------
# WatermarkResult is frozen (immutable)
# ---------------------------------------------------------------------------

def test_watermark_result_frozen():
    result = WatermarkResult(
        num_tokens=10,
        num_green=6,
        green_fraction=0.6,
        z_score=1.26,
        p_value=0.1,
        decision=False,
        threshold=4.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.decision = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _norm_sf edge cases
# ---------------------------------------------------------------------------

def test_norm_sf_at_zero():
    """SF at z=0 should be 0.5."""
    assert abs(_norm_sf(0.0) - 0.5) < 1e-10


def test_norm_sf_large_positive():
    """SF should approach 0 for large positive z."""
    assert _norm_sf(10.0) < 1e-10


def test_norm_sf_large_negative():
    """SF should approach 1 for large negative z."""
    assert _norm_sf(-10.0) > 1.0 - 1e-10


def test_norm_sf_monotone():
    """_norm_sf must be strictly decreasing."""
    zs = [-3.0, -1.0, 0.0, 1.0, 3.0]
    vals = [_norm_sf(z) for z in zs]
    for a, b in zip(vals, vals[1:]):
        assert a > b
