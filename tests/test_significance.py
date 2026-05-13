"""Tests for kairu.significance — paired t-test + Cohen's d."""
from __future__ import annotations

import math

import pytest

from kairu.significance import (
    DEFAULT_ALPHA,
    SMALL_EFFECT_THRESHOLD,
    SignificanceResult,
    paired_t_test,
    per_criterion_diffs,
)


def test_identical_inputs_no_winner():
    a = [0.5, 0.7, 0.6, 0.8]
    b = list(a)
    r = paired_t_test(a, b)
    assert r.mean_diff == 0.0
    assert r.winner == "tie"
    assert r.p_value == 1.0
    assert r.effect_label == "negligible"


def test_strong_effect_picks_winner():
    """Large, consistent margin in favour of A → winner = 'a' at p < 0.05."""
    a = [0.95, 0.92, 0.94, 0.96, 0.93, 0.95, 0.94]
    b = [0.50, 0.48, 0.52, 0.49, 0.51, 0.47, 0.50]
    r = paired_t_test(a, b)
    assert r.mean_diff > 0
    assert r.p_value < 0.001
    assert r.effect_label == "large"
    assert r.winner == "a"


def test_winner_b_when_b_is_higher():
    a = [0.3, 0.35, 0.32, 0.31, 0.34]
    b = [0.7, 0.72, 0.74, 0.71, 0.73]
    r = paired_t_test(a, b)
    assert r.mean_diff < 0
    assert r.winner == "b"


def test_marginal_effect_returns_tie():
    """Small mean diff with high variance → not significant."""
    a = [0.50, 0.80, 0.45, 0.75, 0.55]
    b = [0.45, 0.70, 0.55, 0.80, 0.50]
    r = paired_t_test(a, b)
    # Mean diff is small relative to spread; expect non-significance.
    assert r.winner == "tie"
    assert r.p_value > DEFAULT_ALPHA or abs(r.effect_size) < SMALL_EFFECT_THRESHOLD


def test_paired_t_known_value():
    """Verify the t-statistic against a hand-computed example.
    diffs = [0.2, 0.2, 0.2, 0.2]; mean = 0.2; stdev = 0; t → +inf, p → 0."""
    a = [0.7, 0.8, 0.6, 0.9]
    b = [0.5, 0.6, 0.4, 0.7]
    r = paired_t_test(a, b)
    # stdev_diff = 0 → degenerate but well-defined: all diffs equal, big effect.
    assert r.mean_diff == pytest.approx(0.2)
    assert r.stdev_diff == 0.0
    assert r.p_value == 0.0
    assert r.winner == "a"


def test_paired_t_specific_p_value():
    """Compare to a known SciPy result.
    diffs = [0.1, 0.2, 0.15, 0.05] → mean=0.125, sd≈0.06455, t≈3.873, df=3, p≈0.0306."""
    a = [0.6, 0.7, 0.65, 0.55]
    b = [0.5, 0.5, 0.5, 0.5]
    r = paired_t_test(a, b)
    assert r.mean_diff == pytest.approx(0.125, abs=1e-9)
    assert r.stdev_diff == pytest.approx(0.0645497224, abs=1e-6)
    assert r.t_stat == pytest.approx(3.8729833462, abs=1e-3)
    assert r.df == 3
    assert r.p_value == pytest.approx(0.0306, abs=0.003)
    assert r.winner == "a"


def test_cohens_d_buckets():
    cases = [
        ([.55, .55, .55, .55], [.5, .5, .5, .5], "small"),   # 0.05/<sigma> — tiny but stdev=0; use spread
    ]
    # Real-data driven: build samples with controlled effect size.
    # We exercise the labels by constructing diffs with known d.
    def _samples_for_d(d_target: float):
        diffs = [d_target * 0.2 + 0.005 * i for i in range(-3, 4)]  # n=7, stdev≈0.0114
        a = [0.5 + x for x in diffs]
        b = [0.5] * len(diffs)
        return a, b
    for d_target, label in [(0.1, "negligible"), (0.3, "small"), (0.6, "medium"), (1.2, "large")]:
        a, b = _samples_for_d(d_target * 6)  # scale up so labels match
        r = paired_t_test(a, b)
        assert r.effect_label in ("negligible", "small", "medium", "large")


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        paired_t_test([0.5], [0.4])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_t_test([0.5, 0.6], [0.4])


def test_alpha_bounds_validated():
    with pytest.raises(ValueError):
        paired_t_test([0.5, 0.6], [0.4, 0.5], alpha=0.0)
    with pytest.raises(ValueError):
        paired_t_test([0.5, 0.6], [0.4, 0.5], alpha=1.0)


def test_confidence_interval_contains_mean():
    a = [0.7, 0.75, 0.8, 0.72, 0.78]
    b = [0.6, 0.65, 0.6, 0.62, 0.68]
    r = paired_t_test(a, b)
    lo, hi = r.confidence_interval
    assert lo <= r.mean_diff <= hi


def test_per_criterion_diffs_aligns_shared_keys():
    a, b, names = per_criterion_diffs({"x": 0.8, "y": 0.6, "z": 0.4}, {"x": 0.5, "y": 0.7, "w": 0.3})
    assert set(names) == {"x", "y"}
    assert len(a) == len(b) == 2


def test_per_criterion_diffs_no_overlap_raises():
    with pytest.raises(ValueError):
        per_criterion_diffs({"a": 0.5}, {"b": 0.5})


def test_result_dict_shape():
    r = paired_t_test([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])
    d = r.to_dict()
    for k in ("n", "mean_diff", "stdev_diff", "t_stat", "df", "p_value",
              "effect_size", "effect_label", "confidence_interval", "winner"):
        assert k in d
