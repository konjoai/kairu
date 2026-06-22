"""Tests for kairu.speed_bench — SPEED-Bench semantic task splits."""

from __future__ import annotations

import pytest

from kairu.mock_model import MockModel
from kairu.speed_bench import (
    DEFAULT_SPLITS,
    SpeedBenchReport,
    SplitResult,
    TaskSplit,
    run_speed_bench,
)

# Small, fast splits for the assertions that don't need the full default set.
_SPLITS = (
    TaskSplit("alpha", [1, 2, 3], "first"),
    TaskSplit("beta", [4, 5, 6, 7], "second"),
)


def _run(splits=_SPLITS):
    return run_speed_bench(
        MockModel(), splits=splits, num_tokens=5, num_runs=2, warmup=1
    )


def test_default_splits_cover_six_semantic_tasks():
    names = {s.name for s in DEFAULT_SPLITS}
    assert names == {"translation", "summarization", "qa", "code", "dialogue", "math"}
    assert all(isinstance(s, TaskSplit) and s.prompt for s in DEFAULT_SPLITS)


def test_report_has_one_result_per_split():
    report = _run()
    assert isinstance(report, SpeedBenchReport)
    assert [s.name for s in report.splits] == ["alpha", "beta"]
    assert all(s.tokens_per_s > 0 for s in report.splits)
    assert all(s.mean_s > 0 for s in report.splits)


def test_fastest_and_slowest_are_actual_splits():
    report = _run()
    names = {s.name for s in report.splits}
    assert report.fastest in names
    assert report.slowest in names
    # With ≥2 splits the CV is a non-negative dispersion measure.
    assert report.throughput_cv >= 0.0


def test_single_split_has_zero_dispersion():
    report = _run(splits=(TaskSplit("solo", [1, 2]),))
    assert len(report.splits) == 1
    assert report.throughput_cv == 0.0
    assert report.fastest == "solo"
    assert report.slowest == "solo"


def test_default_splits_used_when_none_passed():
    report = run_speed_bench(MockModel(), num_tokens=4, num_runs=2, warmup=1)
    assert len(report.splits) == len(DEFAULT_SPLITS)


def test_empty_splits_rejected():
    with pytest.raises(ValueError):
        run_speed_bench(MockModel(), splits=())


def test_report_round_trips_to_dict():
    report = _run()
    d = report.as_dict()
    assert set(d) == {
        "splits",
        "fastest",
        "slowest",
        "mean_tokens_per_s",
        "throughput_cv",
    }
    assert d["splits"][0].keys() == {"name", "p50_s", "mean_s", "tokens_per_s"}


def test_split_result_as_dict():
    sr = SplitResult(name="x", p50_s=1.0, mean_s=2.0, tokens_per_s=3.0)
    assert sr.as_dict() == {
        "name": "x",
        "p50_s": 1.0,
        "mean_s": 2.0,
        "tokens_per_s": 3.0,
    }


def test_task_split_description_defaults_empty():
    assert TaskSplit("t", [1]).description == ""
