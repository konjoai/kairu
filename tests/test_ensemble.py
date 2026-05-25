"""Tests for kairu.ensemble — multi-judge evaluation + median aggregation."""

from __future__ import annotations

import statistics

import pytest

from kairu.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    EnsembleComparison,
    EnsembleResult,
    JudgeConfig,
    ensemble_compare,
    ensemble_evaluate,
    judge_evaluate,
)


PROMPT = "Explain photosynthesis briefly with one example."
RESP_GOOD = (
    "Photosynthesis converts light into chemical energy stored as glucose. "
    "Example: oak leaves absorb sunlight via chlorophyll to grow."
)
RESP_BAD = "idk maybe plants do something with sun"


# ── JudgeConfig validation ────────────────────────────────────────────────


def test_judge_config_requires_nonempty_name():
    with pytest.raises(ValueError):
        JudgeConfig(name="")


def test_judge_config_rejects_negative_noise():
    with pytest.raises(ValueError):
        JudgeConfig(name="x", noise=-0.1)


def test_judge_config_rejects_unknown_criteria():
    with pytest.raises(ValueError) as exc:
        JudgeConfig(name="x", criteria=("relevance", "no_such_criterion"))
    assert "no_such_criterion" in str(exc.value)


# ── judge_evaluate ───────────────────────────────────────────────────────


def test_judge_evaluate_deterministic_without_noise():
    j = JudgeConfig(name="j1", rubric="default", noise=0.0)
    r1 = judge_evaluate(PROMPT, RESP_GOOD, j)
    r2 = judge_evaluate(PROMPT, RESP_GOOD, j)
    assert r1.aggregate == r2.aggregate
    assert r1.scores == r2.scores
    assert r1.judge == "j1"
    assert 0.0 <= r1.aggregate <= 1.0


def test_judge_evaluate_with_noise_is_deterministic_per_seed():
    j = JudgeConfig(name="judge-a", noise=0.05)
    r1 = judge_evaluate(PROMPT, RESP_GOOD, j)
    r2 = judge_evaluate(PROMPT, RESP_GOOD, j)
    assert r1.scores == r2.scores
    for k, v in r1.scores.items():
        assert 0.0 <= v <= 1.0


def test_judge_evaluate_noise_changes_with_different_judges():
    j1 = JudgeConfig(name="alpha", noise=0.08)
    j2 = JudgeConfig(name="bravo", noise=0.08)
    r1 = judge_evaluate(PROMPT, RESP_GOOD, j1)
    r2 = judge_evaluate(PROMPT, RESP_GOOD, j2)
    # Different judges → different noise streams → at least one criterion differs.
    assert r1.scores != r2.scores


# ── ensemble_evaluate ────────────────────────────────────────────────────


def test_ensemble_evaluate_requires_judges():
    with pytest.raises(ValueError):
        ensemble_evaluate(PROMPT, RESP_GOOD, [])


def test_ensemble_evaluate_median_is_robust_to_outlier():
    judges = [
        JudgeConfig(name="j1", rubric="default"),
        JudgeConfig(name="j2", rubric="default"),
        JudgeConfig(name="j3", rubric="default"),
        # An outlier judge that only looks at safety (will weight differently)
        JudgeConfig(name="outlier", rubric="safety"),
    ]
    result = ensemble_evaluate(PROMPT, RESP_GOOD, judges)
    assert isinstance(result, EnsembleResult)
    assert len(result.judges) == 4
    # The median should be insulated from the one outlier perspective.
    aggregates = [j.aggregate for j in result.judges]
    assert result.median_aggregate == statistics.median(aggregates)


def test_ensemble_evaluate_disagreement_flag_fires_when_judges_diverge():
    # Wide noise → criteria disagreements above the 0.2 threshold.
    judges = [JudgeConfig(name=f"j{i}", noise=0.35, seed=i) for i in range(5)]
    result = ensemble_evaluate(PROMPT, RESP_GOOD, judges)
    assert result.max_disagreement > 0.0
    # With noise=0.35 and 5 judges, at least one criterion should exceed 0.2.
    assert any(v > 0.2 for v in result.stdev_scores.values())


def test_ensemble_evaluate_no_disagreement_when_judges_identical():
    judges = [JudgeConfig(name="j1"), JudgeConfig(name="j2"), JudgeConfig(name="j3")]
    result = ensemble_evaluate(PROMPT, RESP_GOOD, judges)
    # All judges deterministic with same config → stdev is 0 for every criterion.
    for stdev in result.stdev_scores.values():
        assert stdev == 0.0
    assert result.max_disagreement == 0.0
    assert result.disagreement_flag is False


def test_ensemble_evaluate_to_dict_is_json_serializable():
    import json

    judges = [JudgeConfig(name="j1"), JudgeConfig(name="j2", noise=0.05)]
    result = ensemble_evaluate(PROMPT, RESP_GOOD, judges)
    json.dumps(result.to_dict())  # must not raise
    d = result.to_dict()
    assert d["disagreement_threshold"] == DEFAULT_DISAGREEMENT_THRESHOLD
    assert len(d["judges"]) == 2


# ── ensemble_compare ─────────────────────────────────────────────────────


def test_ensemble_compare_winner_picks_better_response():
    judges = [JudgeConfig(name=f"j{i}") for i in range(3)]
    cmp = ensemble_compare(PROMPT, RESP_GOOD, RESP_BAD, judges)
    assert isinstance(cmp, EnsembleComparison)
    assert cmp.winner == "a"
    assert cmp.median_diff > 0


def test_ensemble_compare_returns_per_criterion_breakdown():
    judges = [JudgeConfig(name="j1"), JudgeConfig(name="j2")]
    cmp = ensemble_compare(PROMPT, RESP_GOOD, RESP_BAD, judges)
    assert set(cmp.per_criterion.keys()) <= set(cmp.a.median_scores.keys())
    for crit, body in cmp.per_criterion.items():
        assert {"a_median", "b_median", "diff", "a_stdev", "b_stdev", "winner"} <= set(
            body
        )
        assert body["winner"] in ("a", "b", "tie")


def test_ensemble_compare_tie_when_responses_match():
    judges = [JudgeConfig(name="j1"), JudgeConfig(name="j2")]
    cmp = ensemble_compare(PROMPT, RESP_GOOD, RESP_GOOD, judges)
    assert cmp.winner == "tie"
    assert abs(cmp.median_diff) < 1e-9


def test_ensemble_compare_propagates_disagreement_flag():
    judges = [JudgeConfig(name=f"j{i}", noise=0.4, seed=i) for i in range(5)]
    cmp = ensemble_compare(PROMPT, RESP_GOOD, RESP_BAD, judges)
    # With this much noise, at least one side should trip the flag.
    assert cmp.disagreement_flag == (cmp.a.disagreement_flag or cmp.b.disagreement_flag)


def test_ensemble_compare_to_dict_is_json_serializable():
    import json

    judges = [JudgeConfig(name="j1"), JudgeConfig(name="j2")]
    cmp = ensemble_compare(PROMPT, RESP_GOOD, RESP_BAD, judges)
    json.dumps(cmp.to_dict())  # no raise
