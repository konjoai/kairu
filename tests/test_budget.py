"""Tests for kairu.budget.TokenBudget — 8 tests."""
import pytest
from kairu.budget import TokenBudget


def test_set_prompt_stores_count():
    b = TokenBudget(max_total_tokens=100)
    b.set_prompt(20)
    assert b.prompt_tokens == 20


def test_set_prompt_negative_raises():
    b = TokenBudget(max_total_tokens=100)
    with pytest.raises(ValueError, match="negative"):
        b.set_prompt(-1)


def test_consume_reduces_remaining():
    b = TokenBudget(max_total_tokens=100)
    b.set_prompt(10)
    consumed = b.consume(5)
    assert consumed == 5
    assert b.remaining == 85
    assert b.generated_tokens == 5


def test_remaining_never_negative():
    b = TokenBudget(max_total_tokens=10)
    b.set_prompt(8)
    # Only 2 tokens remaining; consuming 10 should give back 2
    consumed = b.consume(10)
    assert consumed == 2
    assert b.remaining == 0


def test_exhausted_flag():
    b = TokenBudget(max_total_tokens=5)
    b.set_prompt(3)
    assert not b.exhausted
    b.consume(2)
    assert b.exhausted
    # Further consume returns 0 and stays exhausted
    assert b.consume(1) == 0
    assert b.exhausted


def test_utilization_zero_to_one():
    b = TokenBudget(max_total_tokens=200)
    b.set_prompt(0)
    assert b.utilization() == 0.0
    b.consume(100)
    assert b.utilization() == pytest.approx(0.5)
    b.consume(100)
    assert b.utilization() == pytest.approx(1.0)


def test_reset_generated_clears_counter():
    b = TokenBudget(max_total_tokens=100)
    b.set_prompt(10)
    b.consume(30)
    assert b.generated_tokens == 30
    b.reset_generated()
    assert b.generated_tokens == 0
    assert b.prompt_tokens == 10  # prompt unchanged


def test_total_tokens_is_sum():
    b = TokenBudget(max_total_tokens=200)
    b.set_prompt(15)
    b.consume(7)
    assert b.total_tokens == 22
