"""Tests for kairu.wrapper.ModelWrapper and wrap_model — 8 tests."""
import pytest
from kairu.mock_model import MockModel
from kairu.metrics import GenerationMetrics
from kairu.speculative import SpeculativeDecoder
from kairu.early_exit import EarlyExitDecoder
from kairu.wrapper import ModelWrapper, wrap_model

PROMPT = [10, 20, 30]


def test_wrap_model_with_mock_returns_model_wrapper():
    w = wrap_model(MockModel())
    assert isinstance(w, ModelWrapper)


def test_generate_returns_list_and_metrics():
    w = wrap_model(MockModel())
    tokens, metrics = w.generate(PROMPT, max_new_tokens=10)
    assert isinstance(tokens, list)
    assert isinstance(metrics, GenerationMetrics)


def test_output_length_respects_max_new_tokens():
    w = wrap_model(MockModel())
    for n in (1, 5, 15):
        tokens, _ = w.generate(PROMPT, max_new_tokens=n)
        assert len(tokens) <= n, f"Expected <= {n} tokens, got {len(tokens)}"


def test_budget_limits_output():
    # Prompt is 3 tokens; budget is 8 total → max 5 new tokens
    w = ModelWrapper(MockModel(), max_budget=8)
    tokens, _ = w.generate(PROMPT, max_new_tokens=100)
    assert len(tokens) <= 5, f"Budget exceeded: got {len(tokens)} tokens"


def test_metrics_generated_tokens_matches_output():
    w = wrap_model(MockModel())
    tokens, metrics = w.generate(PROMPT, max_new_tokens=10)
    assert metrics.generated_tokens == len(tokens)


def test_metrics_total_time_positive():
    w = wrap_model(MockModel())
    _, metrics = w.generate(PROMPT, max_new_tokens=5)
    assert metrics.total_time_ms > 0.0


def test_with_draft_model_uses_speculative_decoder():
    w = ModelWrapper(
        model=MockModel(),
        draft_model=MockModel(),
        speculative_gamma=3,
    )
    assert isinstance(w._decoder, SpeculativeDecoder)
    tokens, metrics = w.generate(PROMPT, max_new_tokens=10)
    assert isinstance(tokens, list)
    # Speculative stats populate accepted/rejected
    assert metrics.accepted_tokens >= 0
    assert metrics.rejected_tokens >= 0


def test_wrap_model_string_falls_back_to_mock():
    # Without torch/transformers installed, loading by name must silently
    # fall back to MockModel and return a working ModelWrapper.
    w = wrap_model("some-nonexistent-model-name")
    assert isinstance(w, ModelWrapper)
    tokens, metrics = w.generate([1, 2, 3], max_new_tokens=5)
    assert isinstance(tokens, list)
    assert metrics.generated_tokens == len(tokens)
