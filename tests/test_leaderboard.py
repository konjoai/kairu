"""Tests for kairu.leaderboard — SQLite-backed model score history."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from kairu.leaderboard import (
    LeaderboardEntry,
    LeaderboardRow,
    LeaderboardStore,
)


def _iso(offset_seconds: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def test_record_then_rank_basic():
    store = LeaderboardStore()
    store.record(model="alpha", prompt="x", rubric_name="default", aggregate=0.8)
    store.record(model="alpha", prompt="y", rubric_name="default", aggregate=0.9)
    store.record(model="beta",  prompt="x", rubric_name="default", aggregate=0.5)
    ranking = store.rank(metric="aggregate", limit=10)
    assert len(ranking) == 2
    assert ranking[0].model == "alpha"
    assert ranking[0].rank == 1
    assert ranking[0].mean_score == pytest.approx(0.85, abs=1e-9)
    assert ranking[1].model == "beta"
    assert ranking[1].n_evaluations == 1


def test_record_rejects_bad_aggregate():
    store = LeaderboardStore()
    with pytest.raises(ValueError):
        store.record(model="alpha", prompt="x", rubric_name="default", aggregate=1.5)
    with pytest.raises(ValueError):
        store.record(model="alpha", prompt="x", rubric_name="default", aggregate=-0.1)


def test_record_rejects_empty_model_name():
    store = LeaderboardStore()
    with pytest.raises(ValueError):
        store.record(model="", prompt="x", rubric_name="default", aggregate=0.5)
    with pytest.raises(ValueError):
        store.record(model="   ", prompt="x", rubric_name="default", aggregate=0.5)


def test_rank_by_criterion_filters_missing_rows():
    store = LeaderboardStore()
    store.record(model="alpha", prompt="x", rubric_name="default", aggregate=0.5,
                 criteria={"relevance": 0.9, "fluency": 0.8})
    store.record(model="beta",  prompt="x", rubric_name="default", aggregate=0.6,
                 criteria={"relevance": 0.4})  # no fluency
    ranking = store.rank(metric="fluency", limit=10)
    # Only alpha has 'fluency' — beta is excluded for that metric.
    assert len(ranking) == 1
    assert ranking[0].model == "alpha"
    assert ranking[0].mean_score == pytest.approx(0.8, abs=1e-9)


def test_rank_days_window_filters_old_entries():
    store = LeaderboardStore()
    # Old entry — 30 days ago.
    store.record(model="alpha", prompt="x", rubric_name="default", aggregate=0.9,
                 timestamp_utc=_iso(-30 * 86400))
    # Recent entry — 1 day ago.
    store.record(model="alpha", prompt="y", rubric_name="default", aggregate=0.5,
                 timestamp_utc=_iso(-1 * 86400))
    seven_day = store.rank(metric="aggregate", days=7)
    assert seven_day[0].mean_score == pytest.approx(0.5, abs=1e-9)
    all_time = store.rank(metric="aggregate", days=None)
    assert all_time[0].mean_score == pytest.approx(0.7, abs=1e-9)


def test_delta_positive_on_improvement():
    store = LeaderboardStore()
    # Prior period (15-12 days ago): low scores
    for off in (-15, -14, -13, -12):
        store.record(model="m", prompt=f"p{off}", rubric_name="default", aggregate=0.4,
                     timestamp_utc=_iso(off * 86400))
    # Current period (last 7d): high scores
    for off in (-6, -5, -4, -3, -2, -1):
        store.record(model="m", prompt=f"q{off}", rubric_name="default", aggregate=0.9,
                     timestamp_utc=_iso(off * 86400))
    ranking = store.rank(metric="aggregate", days=7)
    assert ranking[0].delta == pytest.approx(0.5, abs=1e-9)


def test_delta_zero_when_no_prior_data():
    store = LeaderboardStore()
    store.record(model="m", prompt="x", rubric_name="default", aggregate=0.8)
    ranking = store.rank(metric="aggregate", days=7)
    assert ranking[0].delta == 0.0


def test_trend_returns_last_ten_scores_in_order():
    store = LeaderboardStore()
    for i in range(12):
        store.record(model="m", prompt=f"p{i}", rubric_name="default",
                     aggregate=i / 11.0,
                     timestamp_utc=_iso(-(12 - i)))  # ascending in time
    ranking = store.rank(metric="aggregate")
    trend = ranking[0].trend
    assert len(trend) == 10
    # Trend is chronological — last entry has the highest score.
    assert trend[-1] > trend[0]


def test_rank_limit_caps_results():
    store = LeaderboardStore()
    for n in range(25):
        store.record(model=f"m{n}", prompt="x", rubric_name="default",
                     aggregate=n / 24.0)
    ranking = store.rank(metric="aggregate", limit=5)
    assert len(ranking) == 5
    # Best model should be on top.
    assert ranking[0].model == "m24"


def test_rank_rejects_bad_limit():
    store = LeaderboardStore()
    with pytest.raises(ValueError):
        store.rank(metric="aggregate", limit=0)


def test_percentiles_present_in_ranking():
    store = LeaderboardStore()
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        store.record(model="m", prompt="x", rubric_name="default", aggregate=s)
    row = store.rank(metric="aggregate")[0]
    assert 0.0 <= row.p25 <= row.p50 <= row.p75 <= 1.0


def test_list_models_returns_sorted_unique():
    store = LeaderboardStore()
    store.record(model="zebra", prompt="x", rubric_name="default", aggregate=0.5)
    store.record(model="alpha", prompt="x", rubric_name="default", aggregate=0.5)
    store.record(model="alpha", prompt="y", rubric_name="default", aggregate=0.5)
    assert store.list_models() == ["alpha", "zebra"]


def test_to_dict_round_trips_through_json():
    store = LeaderboardStore()
    store.record(model="m", prompt="x", rubric_name="default", aggregate=0.7,
                 criteria={"a": 0.5, "b": 0.6})
    row = store.rank(metric="aggregate")[0]
    encoded = json.dumps(row.to_dict())
    assert "mean_score" in json.loads(encoded)
