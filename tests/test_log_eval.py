"""Tests for kairu.log_eval — production log → evaluation pipeline."""
from __future__ import annotations

import pytest

from kairu.log_eval import (
    DEFAULT_LOG_THRESHOLD,
    LogEvalReport,
    LogItemResult,
    evaluate_log,
)


LOG_GOOD = [
    {"input": "Define recursion",
     "output": "Recursion is when a function calls itself with a smaller input until it hits a base case.",
     "request_id": "req-1", "region": "us-east-1"},
    {"input": "What is the speed of light?",
     "output": "Light travels at approximately 299,792 kilometres per second in a vacuum.",
     "request_id": "req-2"},
    {"input": "Summarize the water cycle",
     "output": "Water evaporates from oceans, condenses into clouds, and falls as precipitation, then flows back to the oceans.",
     "request_id": "req-3"},
]

LOG_BAD = [
    {"input": "Define recursion", "output": "no idea"},
    {"input": "What is the speed of light?", "output": "fast"},
    {"input": "Summarize the water cycle", "output": "h2o moves"},
]


def test_evaluate_log_rejects_empty():
    with pytest.raises(ValueError):
        evaluate_log([])


def test_evaluate_log_rejects_bad_threshold():
    with pytest.raises(ValueError):
        evaluate_log(LOG_GOOD, threshold=1.5)
    with pytest.raises(ValueError):
        evaluate_log(LOG_GOOD, threshold=-0.1)


def test_evaluate_log_rejects_non_string_fields():
    with pytest.raises(TypeError):
        evaluate_log([{"input": 1, "output": "x"}])
    with pytest.raises(TypeError):
        evaluate_log([{"input": "x", "output": None}])


def test_evaluate_log_returns_per_item_results():
    report = evaluate_log(LOG_GOOD, rubric="default")
    assert isinstance(report, LogEvalReport)
    assert report.n_items == 3
    assert len(report.items) == 3
    for item in report.items:
        assert isinstance(item, LogItemResult)
        assert 0.0 <= item.aggregate <= 1.0
        assert isinstance(item.passed, bool)


def test_evaluate_log_preserves_metadata():
    report = evaluate_log(LOG_GOOD)
    assert report.items[0].metadata.get("request_id") == "req-1"
    assert report.items[0].metadata.get("region") == "us-east-1"
    assert report.items[1].metadata.get("request_id") == "req-2"


def test_evaluate_log_aggregates_are_consistent_with_items():
    report = evaluate_log(LOG_GOOD)
    aggs = [it.aggregate for it in report.items]
    assert report.mean_aggregate == pytest.approx(sum(aggs) / len(aggs), rel=1e-9)
    assert report.min_aggregate == min(aggs)
    assert report.max_aggregate == max(aggs)


def test_evaluate_log_pass_flag_tracks_threshold():
    high = evaluate_log(LOG_GOOD, threshold=0.05)
    assert high.passed is True
    low = evaluate_log(LOG_BAD, threshold=0.95)
    assert low.passed is False
    assert low.n_failed >= 1


def test_evaluate_log_per_criterion_means_match_columns():
    report = evaluate_log(LOG_GOOD)
    for crit, mean_v in report.per_criterion_mean.items():
        col = [it.scores[crit] for it in report.items if crit in it.scores]
        assert mean_v == pytest.approx(sum(col) / len(col), rel=1e-9)
        assert report.per_criterion_min[crit] == min(col)


def test_evaluate_log_to_dict_is_json_serializable():
    import json
    report = evaluate_log(LOG_GOOD)
    json.dumps(report.to_dict())  # must not raise
    d = report.to_dict()
    assert d["n_items"] == 3
    assert "duration_seconds" in d


def test_evaluate_log_threshold_default():
    report = evaluate_log(LOG_GOOD)
    assert report.threshold == DEFAULT_LOG_THRESHOLD


def test_evaluate_log_n_failed_counts_correctly():
    report = evaluate_log(LOG_BAD, threshold=0.99)
    assert report.n_failed == 3
    report2 = evaluate_log(LOG_BAD, threshold=0.0)
    assert report2.n_failed == 0
