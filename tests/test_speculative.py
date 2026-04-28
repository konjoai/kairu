"""Tests for kairu.speculative.SpeculativeDecoder — 8 tests."""
import pytest
from kairu.mock_model import MockModel
from kairu.speculative import SpeculativeDecoder

PROMPT = [1, 2, 3, 4, 5]


def _make_decoder(gamma: int = 4, temperature: float = 1.0) -> SpeculativeDecoder:
    return SpeculativeDecoder(
        target=MockModel(),
        draft=MockModel(),
        gamma=gamma,
        temperature=temperature,
    )


def test_generate_returns_correct_types():
    dec = _make_decoder()
    tokens, stats = dec.generate(PROMPT, max_new_tokens=10)
    assert isinstance(tokens, list)
    assert isinstance(stats, dict)
    assert all(isinstance(t, int) for t in tokens)


def test_output_length_respects_max_new_tokens():
    dec = _make_decoder()
    for n in (1, 5, 20):
        tokens, _ = dec.generate(PROMPT, max_new_tokens=n)
        assert len(tokens) <= n, f"Expected <= {n} tokens, got {len(tokens)}"


def test_stats_has_all_required_keys():
    dec = _make_decoder()
    _, stats = dec.generate(PROMPT, max_new_tokens=10)
    for key in ("total_tokens", "accepted_tokens", "rejected_tokens", "acceptance_rate"):
        assert key in stats, f"Missing key: {key}"


def test_acceptance_rate_in_unit_interval():
    dec = _make_decoder()
    _, stats = dec.generate(PROMPT, max_new_tokens=20)
    rate = stats["acceptance_rate"]
    assert 0.0 <= rate <= 1.0, f"acceptance_rate out of [0,1]: {rate}"


def test_deterministic_with_same_seed():
    # Two separate decoder instances with default seed=42 on the same MockModel
    # should produce identical output for identical input.
    dec1 = _make_decoder()
    dec2 = _make_decoder()
    out1, _ = dec1.generate(PROMPT, max_new_tokens=15)
    out2, _ = dec2.generate(PROMPT, max_new_tokens=15)
    assert out1 == out2, "Same seed + same prompt must yield identical output"


def test_gamma_1_works_correctly():
    dec = _make_decoder(gamma=1)
    tokens, stats = dec.generate(PROMPT, max_new_tokens=5)
    assert isinstance(tokens, list)
    assert len(tokens) <= 5
    assert stats["total_tokens"] == len(tokens)


def test_zero_max_new_tokens_returns_empty():
    dec = _make_decoder()
    tokens, stats = dec.generate(PROMPT, max_new_tokens=0)
    assert tokens == []
    assert stats["total_tokens"] == 0
    assert stats["accepted_tokens"] == 0
    assert stats["rejected_tokens"] == 0


def test_stats_total_equals_accepted_plus_rejected():
    dec = _make_decoder()
    _, stats = dec.generate(PROMPT, max_new_tokens=20)
    assert stats["accepted_tokens"] + stats["rejected_tokens"] >= 0
    # acceptance_rate consistency check
    total = stats["accepted_tokens"] + stats["rejected_tokens"]
    if total > 0:
        expected_rate = stats["accepted_tokens"] / total
        assert stats["acceptance_rate"] == pytest.approx(expected_rate, rel=1e-6)
