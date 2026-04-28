"""Tests for kairu.metrics.GenerationMetrics — 7 tests."""
import time
import pytest
from kairu.metrics import GenerationMetrics


def test_initial_state():
    m = GenerationMetrics(prompt_tokens=8)
    assert m.prompt_tokens == 8
    assert m.generated_tokens == 0
    assert m.accepted_tokens == 0
    assert m.rejected_tokens == 0


def test_record_token_increments_count():
    m = GenerationMetrics()
    assert m.generated_tokens == 0
    m.record_token()
    m.record_token()
    assert m.generated_tokens == 2


def test_finish_seals_measurement():
    m = GenerationMetrics()
    m.record_token()
    m.finish()
    t1 = m.total_time_ms
    time.sleep(0.02)
    t2 = m.total_time_ms
    # After finish(), total_time_ms must not grow
    assert t2 == pytest.approx(t1, abs=0.5)


def test_tokens_per_second_positive_after_generation():
    m = GenerationMetrics()
    for _ in range(10):
        m.record_token()
    m.finish()
    assert m.tokens_per_second > 0.0


def test_acceptance_rate_zero_when_no_speculative():
    m = GenerationMetrics()
    # No accepted/rejected counters set → rate must be 0
    assert m.acceptance_rate == 0.0


def test_to_dict_has_required_keys():
    m = GenerationMetrics(prompt_tokens=4)
    m.record_token()
    m.finish()
    d = m.to_dict()
    required = {
        "prompt_tokens",
        "generated_tokens",
        "total_time_ms",
        "tokens_per_second",
        "mean_latency_ms",
        "acceptance_rate",
    }
    assert required.issubset(d.keys())


def test_mean_latency_ms_proportional():
    m = GenerationMetrics()
    for _ in range(5):
        m.record_token()
    m.finish()
    # mean_latency = total_time / generated_tokens
    assert m.mean_latency_ms == pytest.approx(
        m.total_time_ms / m.generated_tokens, rel=1e-6
    )
