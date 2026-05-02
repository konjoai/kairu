"""Tests for kairu.layered — per-layer early exit."""
from __future__ import annotations

import numpy as np
import pytest

from kairu.layered import (
    LayeredModelInterface,
    LayerwiseEarlyExitDecoder,
    MockLayeredModel,
)


def test_mock_layered_model_implements_interface():
    m = MockLayeredModel(num_layers=8)
    assert isinstance(m, LayeredModelInterface)
    assert m.num_layers() == 8
    assert m.vocab_size == 1000


def test_layer_logits_shape_and_dtype():
    m = MockLayeredModel(num_layers=4)
    logits = m.layer_logits([1, 2, 3], 2)
    assert logits.shape == (m.vocab_size,)
    assert logits.dtype == np.float32


def test_layer_logits_rejects_out_of_range():
    m = MockLayeredModel(num_layers=4)
    with pytest.raises(ValueError):
        m.layer_logits([1], 0)
    with pytest.raises(ValueError):
        m.layer_logits([1], 5)


def test_layer_logits_sharpen_with_depth():
    """Final layer must be at least as confident as any earlier layer."""
    m = MockLayeredModel(num_layers=12)
    tokens = [10, 20, 30]
    confidences = []
    for layer in range(1, m.num_layers() + 1):
        logits = m.layer_logits(tokens, layer)
        x = logits - logits.max()
        probs = np.exp(x) / np.exp(x).sum()
        confidences.append(probs.max())
    assert confidences[-1] >= confidences[0]


def test_next_token_logits_matches_final_layer():
    m = MockLayeredModel(num_layers=6)
    final = m.layer_logits([1, 2], 6)
    np.testing.assert_array_equal(m.next_token_logits([1, 2]), final)


def test_decoder_generates_requested_tokens():
    m = MockLayeredModel(num_layers=8)
    dec = LayerwiseEarlyExitDecoder(m, confidence_threshold=0.5)
    out, stats = dec.generate([1, 2, 3], max_new_tokens=5)
    assert stats["tokens_generated"] == 5
    assert len(out) == 5
    assert len(stats["exit_layers"]) == 5
    assert stats["total_layers"] == 8


def test_decoder_exit_layers_within_bounds():
    m = MockLayeredModel(num_layers=8)
    dec = LayerwiseEarlyExitDecoder(m, confidence_threshold=0.95, min_layer=2)
    out, stats = dec.generate([1, 2, 3], max_new_tokens=4)
    for layer in stats["exit_layers"]:
        assert 2 <= layer <= 8


def test_decoder_compute_saved_high_with_low_threshold():
    """Low threshold → exits at min_layer for nearly all tokens → compute_saved high."""
    m = MockLayeredModel(num_layers=12)
    dec = LayerwiseEarlyExitDecoder(m, confidence_threshold=1e-6, min_layer=1)
    _, stats = dec.generate([1], max_new_tokens=4)
    # Threshold 1e-6 always satisfied at layer 1 → mean_exit ≈ 1, saved ≈ 11/12
    assert stats["mean_exit_layer"] == 1
    assert stats["compute_saved"] > 0.9


def test_decoder_compute_saved_zero_with_unreachable_threshold():
    """Threshold 1.0 (unreachable for non-degenerate softmax) → all tokens go to layer L."""
    m = MockLayeredModel(num_layers=8)
    dec = LayerwiseEarlyExitDecoder(m, confidence_threshold=1.0)
    _, stats = dec.generate([1], max_new_tokens=3)
    assert all(layer == 8 for layer in stats["exit_layers"])
    assert stats["compute_saved"] == 0.0


def test_decoder_rejects_bad_config():
    m = MockLayeredModel(num_layers=4)
    with pytest.raises(ValueError):
        LayerwiseEarlyExitDecoder(m, confidence_threshold=0.0)
    with pytest.raises(ValueError):
        LayerwiseEarlyExitDecoder(m, min_layer=0)
    dec = LayerwiseEarlyExitDecoder(m, min_layer=10)
    with pytest.raises(ValueError):
        dec.generate([1], max_new_tokens=1)


def test_mock_layered_rejects_zero_layers():
    with pytest.raises(ValueError):
        MockLayeredModel(num_layers=0)
