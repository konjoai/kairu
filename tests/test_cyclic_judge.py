"""Tests for kairu.cyclic_judge — round-robin allocation, coverage-correct
batch intervals, and judge/item variance decomposition."""

from __future__ import annotations

import statistics

import pytest

from kairu.cyclic_judge import (
    CyclicEvalReport,
    JudgeLoad,
    MeanInterval,
    batch_mean_interval,
    cyclic_allocate,
    cyclic_evaluate,
    full_grid_scores,
    variance_components,
)
from kairu.ensemble import JudgeConfig

PROMPT = "Explain photosynthesis briefly with one example."
RESP_GOOD = (
    "Photosynthesis converts light into chemical energy stored as glucose. "
    "Example: oak leaves absorb sunlight via chlorophyll to grow."
)
RESP_BAD = "idk maybe plants do something with sun"

JUDGES = [
    JudgeConfig(name="alpha", noise=0.05, seed=1),
    JudgeConfig(name="beta", noise=0.05, seed=2),
    JudgeConfig(name="gamma", noise=0.05, seed=3),
]

ITEMS = [
    (PROMPT, RESP_GOOD),
    (PROMPT, RESP_BAD),
    ("Define entropy.", "Entropy measures disorder in a system."),
    ("What is 2+2?", "Four."),
    ("Summarise the water cycle.", "Water evaporates, condenses, and falls as rain."),
]


# ── cyclic_allocate ────────────────────────────────────────────────────────


def test_cyclic_allocate_rotates_round_robin():
    assert cyclic_allocate(5, 3) == [0, 1, 2, 0, 1]


def test_cyclic_allocate_respects_offset():
    assert cyclic_allocate(4, 3, offset=2) == [2, 0, 1, 2]


def test_cyclic_allocate_empty_items():
    assert cyclic_allocate(0, 3) == []


def test_cyclic_allocate_rejects_zero_judges():
    with pytest.raises(ValueError):
        cyclic_allocate(5, 0)


def test_cyclic_allocate_rejects_negative_items():
    with pytest.raises(ValueError):
        cyclic_allocate(-1, 3)


def test_cyclic_allocate_balance_within_one():
    alloc = cyclic_allocate(10, 3)
    counts = [alloc.count(j) for j in range(3)]
    assert max(counts) - min(counts) <= 1


# ── batch_mean_interval ────────────────────────────────────────────────────


def test_batch_mean_interval_brackets_mean():
    iv = batch_mean_interval([0.4, 0.5, 0.6, 0.5, 0.55])
    assert iv.lo < iv.mean < iv.hi
    assert iv.mean == pytest.approx(statistics.fmean([0.4, 0.5, 0.6, 0.5, 0.55]))
    assert iv.n == 5
    assert iv.confidence == 0.95


def test_batch_mean_interval_single_item_is_degenerate():
    iv = batch_mean_interval([0.7])
    assert iv.lo == iv.hi == iv.mean == 0.7
    assert iv.stdev == 0.0


def test_batch_mean_interval_widens_with_variance():
    tight = batch_mean_interval([0.5, 0.5, 0.5, 0.51, 0.49])
    wide = batch_mean_interval([0.1, 0.9, 0.2, 0.8, 0.5])
    assert (wide.hi - wide.lo) > (tight.hi - tight.lo)


def test_batch_mean_interval_higher_confidence_is_wider():
    lo95 = batch_mean_interval([0.2, 0.4, 0.6, 0.8], confidence=0.95)
    lo99 = batch_mean_interval([0.2, 0.4, 0.6, 0.8], confidence=0.99)
    assert (lo99.hi - lo99.lo) > (lo95.hi - lo95.lo)


def test_batch_mean_interval_rejects_empty():
    with pytest.raises(ValueError):
        batch_mean_interval([])


def test_batch_mean_interval_rejects_bad_confidence():
    with pytest.raises(ValueError):
        batch_mean_interval([0.1, 0.2], confidence=1.5)


def test_mean_interval_to_dict_round_trips():
    iv = batch_mean_interval([0.3, 0.6, 0.9])
    d = iv.to_dict()
    assert d["n"] == 3
    assert set(d) == {"mean", "lo", "hi", "n", "stdev", "confidence"}


# ── variance_components ────────────────────────────────────────────────────


def test_variance_components_pure_item_signal():
    # Judges agree perfectly; all variance is item signal, none is judge bias.
    grid = {"a": [0.2, 0.8, 0.5], "b": [0.2, 0.8, 0.5]}
    vc = variance_components(grid)
    assert vc.judge_variance_fraction == pytest.approx(0.0)
    assert vc.item_variance_fraction == pytest.approx(1.0)
    assert vc.ss_residual == pytest.approx(0.0)


def test_variance_components_pure_judge_bias():
    # Items identical; every difference is a constant per-judge offset.
    grid = {"a": [0.3, 0.3, 0.3], "b": [0.7, 0.7, 0.7]}
    vc = variance_components(grid)
    assert vc.judge_variance_fraction == pytest.approx(1.0)
    assert vc.item_variance_fraction == pytest.approx(0.0)


def test_variance_components_identity_holds():
    grid = {"a": [0.1, 0.6, 0.4], "b": [0.5, 0.7, 0.2], "c": [0.3, 0.9, 0.5]}
    vc = variance_components(grid)
    assert vc.ss_total == pytest.approx(vc.ss_judge + vc.ss_item + vc.ss_residual)


def test_variance_components_all_equal_is_zero():
    grid = {"a": [0.5, 0.5], "b": [0.5, 0.5]}
    vc = variance_components(grid)
    assert vc.ss_total == pytest.approx(0.0)
    assert vc.judge_variance_fraction == 0.0
    assert vc.item_variance_fraction == 0.0


def test_variance_components_rejects_single_judge():
    with pytest.raises(ValueError):
        variance_components({"a": [0.1, 0.2]})


def test_variance_components_rejects_ragged_grid():
    with pytest.raises(ValueError):
        variance_components({"a": [0.1, 0.2], "b": [0.3]})


def test_variance_components_rejects_single_item():
    with pytest.raises(ValueError):
        variance_components({"a": [0.1], "b": [0.2]})


def test_variance_components_to_dict():
    vc = variance_components({"a": [0.1, 0.6], "b": [0.5, 0.7]})
    d = vc.to_dict()
    assert d["n_judges"] == 2
    assert d["n_items"] == 2


# ── cyclic_evaluate ────────────────────────────────────────────────────────


def test_cyclic_evaluate_one_judge_per_item():
    report = cyclic_evaluate(ITEMS, JUDGES)
    assert isinstance(report, CyclicEvalReport)
    assert len(report.per_item) == len(ITEMS)
    assert len(report.assignments) == len(ITEMS)
    # Round-robin assignment over three judges.
    assert [a[1] for a in report.assignments] == [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
    ]


def test_cyclic_evaluate_balance_within_one():
    report = cyclic_evaluate(ITEMS, JUDGES)
    assert report.balance <= 1
    assert all(isinstance(load, JudgeLoad) for load in report.judge_loads)


def test_cyclic_evaluate_mean_matches_per_item():
    report = cyclic_evaluate(ITEMS, JUDGES)
    expected = statistics.fmean([s.aggregate for s in report.per_item])
    assert report.mean_aggregate == pytest.approx(expected)
    assert report.interval.mean == pytest.approx(expected)


def test_cyclic_evaluate_interval_is_coverage_unit_items():
    report = cyclic_evaluate(ITEMS, JUDGES)
    assert isinstance(report.interval, MeanInterval)
    assert report.interval.n == len(ITEMS)


def test_cyclic_evaluate_deterministic():
    a = cyclic_evaluate(ITEMS, JUDGES)
    b = cyclic_evaluate(ITEMS, JUDGES)
    assert a.mean_aggregate == b.mean_aggregate
    assert a.to_dict() == b.to_dict()


def test_cyclic_evaluate_offset_changes_assignment():
    base = cyclic_evaluate(ITEMS, JUDGES, offset=0)
    rot = cyclic_evaluate(ITEMS, JUDGES, offset=1)
    assert [a[1] for a in base.assignments] != [a[1] for a in rot.assignments]


def test_cyclic_evaluate_more_judges_than_items():
    report = cyclic_evaluate(ITEMS[:2], JUDGES)
    # Two items, three judges → one judge sees zero items.
    idle = [load for load in report.judge_loads if load.n_items == 0]
    assert len(idle) == 1
    assert idle[0].mean_aggregate == 0.0


def test_cyclic_evaluate_rejects_empty_items():
    with pytest.raises(ValueError):
        cyclic_evaluate([], JUDGES)


def test_cyclic_evaluate_rejects_empty_judges():
    with pytest.raises(ValueError):
        cyclic_evaluate(ITEMS, [])


def test_cyclic_evaluate_to_dict_shape():
    d = cyclic_evaluate(ITEMS, JUDGES).to_dict()
    assert set(d) == {
        "assignments",
        "per_item",
        "judge_loads",
        "mean_aggregate",
        "balance",
        "interval",
    }


# ── full_grid_scores ───────────────────────────────────────────────────────


def test_full_grid_scores_shape():
    grid = full_grid_scores(ITEMS, JUDGES)
    assert set(grid) == {"alpha", "beta", "gamma"}
    assert all(len(row) == len(ITEMS) for row in grid.values())


def test_full_grid_feeds_variance_components():
    grid = full_grid_scores(ITEMS, JUDGES)
    vc = variance_components(grid)
    assert 0.0 <= vc.judge_variance_fraction <= 1.0
    assert vc.n_items == len(ITEMS)
    assert vc.n_judges == len(JUDGES)


def test_full_grid_scores_rejects_empty_items():
    with pytest.raises(ValueError):
        full_grid_scores([], JUDGES)


def test_full_grid_scores_rejects_empty_judges():
    with pytest.raises(ValueError):
        full_grid_scores(ITEMS, [])


def test_full_grid_zero_noise_judges_agree():
    # Deterministic judges (noise=0) produce identical rows → no judge variance.
    judges = [JudgeConfig(name="x"), JudgeConfig(name="y")]
    grid = full_grid_scores(ITEMS, judges)
    vc = variance_components(grid)
    assert vc.judge_variance_fraction == pytest.approx(0.0, abs=1e-9)
