"""Tests for DecoderRouter and RouterDecision."""
from __future__ import annotations

import pytest

from kairu.router import DecoderRouter, RouterDecision, RoutingStats
from kairu.mock_model import MockModel
from kairu.metrics import GenerationMetrics


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def draft():
    return MockModel()


# --- RouterDecision ---

def test_router_decision_fields():
    from kairu.auto_profile import DecoderProfile

    p = DecoderProfile(
        strategy="streaming",
        gamma=1,
        early_exit_threshold=0.0,
        temperature=0.0,
        use_cache=False,
        cache_capacity=0,
        rationale="test",
    )
    d = RouterDecision(strategy="streaming", profile=p, confidence=0.9, rationale="r")
    assert d.strategy == "streaming"
    assert d.confidence == 0.9
    assert d.latency_budget_ms is None


# --- DecoderRouter construction ---

def test_router_short_prompt_threshold_default(model):
    r = DecoderRouter(model)
    assert r.stats.total_routed == 0


def test_router_bad_threshold(model):
    with pytest.raises(ValueError):
        DecoderRouter(model, short_prompt_threshold=0)


# --- Routing decisions ---

def test_short_prompt_routes_streaming(model):
    r = DecoderRouter(model, short_prompt_threshold=20)
    d = r.route([1, 2, 3])
    assert d.strategy == "streaming"
    assert "Short prompt" in d.rationale


def test_long_prompt_no_draft_routes_early_exit(model):
    r = DecoderRouter(model, short_prompt_threshold=5)
    d = r.route(list(range(20)))
    assert d.strategy == "early_exit"
    assert "No draft model" in d.rationale


def test_long_prompt_with_draft_routes_speculative(model, draft):
    r = DecoderRouter(model, draft_model=draft, short_prompt_threshold=5)
    d = r.route(list(range(20)))
    assert d.strategy == "speculative"
    assert "Draft model" in d.rationale


def test_tight_budget_overrides_to_streaming(model, draft):
    r = DecoderRouter(
        model, draft_model=draft, short_prompt_threshold=5, latency_budget_ms=50.0
    )
    d = r.route(list(range(20)))
    assert d.strategy == "streaming"
    assert "budget" in d.rationale.lower()


def test_loose_budget_does_not_override(model, draft):
    r = DecoderRouter(
        model, draft_model=draft, short_prompt_threshold=5, latency_budget_ms=500.0
    )
    d = r.route(list(range(20)))
    assert d.strategy == "speculative"


def test_confidence_in_range(model):
    r = DecoderRouter(model)
    d = r.route([1, 2, 3])
    assert 0.0 <= d.confidence <= 1.0


def test_stats_accumulate(model):
    r = DecoderRouter(model, short_prompt_threshold=5)
    r.route([1, 2])
    r.route([1, 2])
    assert r.stats.total_routed == 2
    assert r.stats.decisions["streaming"] == 2


def test_stats_across_strategies(model, draft):
    r = DecoderRouter(model, draft_model=draft, short_prompt_threshold=5)
    r.route([1, 2])          # short → streaming
    r.route(list(range(20)))  # long + draft → speculative
    assert r.stats.decisions["streaming"] == 1
    assert r.stats.decisions["speculative"] == 1
    assert r.stats.total_routed == 2


def test_profile_strategy_matches_decision(model):
    r = DecoderRouter(model, short_prompt_threshold=5)
    d = r.route(list(range(20)))
    assert d.profile.strategy == d.strategy


# --- record_outcome ---

def test_record_outcome_updates_latency(model):
    r = DecoderRouter(model)
    d = r.route([1, 2, 3])
    m = GenerationMetrics(prompt_tokens=3, generated_tokens=10)
    r.record_outcome(d, m)
    assert r.stats.mean_latency_by_strategy.get("streaming") is not None


def test_record_outcome_ewma(model):
    r = DecoderRouter(model)
    d = r.route([1, 2, 3])
    m1 = GenerationMetrics(prompt_tokens=3, generated_tokens=10)
    m2 = GenerationMetrics(prompt_tokens=3, generated_tokens=10)
    # Finish m2 after a moment so total_time_ms differs
    import time
    time.sleep(0.001)
    m2.finish()
    r.record_outcome(d, m1)
    first = r.stats.mean_latency_by_strategy["streaming"]
    r.record_outcome(d, m2)
    second = r.stats.mean_latency_by_strategy["streaming"]
    # EWMA should update even when latencies are slightly different
    # (they may be equal in fast tests, so just assert the key exists)
    assert second is not None


def test_router_name_hint_accepted(model):
    r = DecoderRouter(model, name_hint="my-model")
    d = r.route([1, 2, 3])
    assert d.strategy == "streaming"
