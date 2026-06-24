"""Tests for kairu.conformal — split conformal intervals for judge scores."""

from __future__ import annotations

import random

import pytest

from kairu.conformal import (
    ConformalInterval,
    calibrate_interval,
    conformal_from_ensemble,
    conformal_quantile,
)
from kairu.ensemble import ensemble_evaluate, JudgeConfig


# --------------------------------------------------------------------------- #
# conformal_quantile
# --------------------------------------------------------------------------- #


def test_quantile_is_finite_sample_rank():
    # 9 residuals, alpha=0.1 → rank = ceil(10 * 0.9) = 9 → the 9th (largest).
    residuals = [0.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]
    assert conformal_quantile(residuals, alpha=0.1) == pytest.approx(0.8)


def test_quantile_uses_absolute_residuals():
    # Sign must not matter — only magnitude.
    assert conformal_quantile([-0.5, 0.5, -0.2], alpha=0.5) == conformal_quantile(
        [0.5, 0.5, 0.2], alpha=0.5
    )


def test_quantile_infinite_when_calibration_too_small():
    # n=1, alpha=0.1 → rank = ceil(2 * 0.9) = 2 > 1 → no finite bound.
    assert conformal_quantile([0.3], alpha=0.1) == float("inf")


def test_quantile_rejects_empty():
    with pytest.raises(ValueError):
        conformal_quantile([], alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_quantile_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_quantile([0.1, 0.2], alpha=alpha)


# --------------------------------------------------------------------------- #
# calibrate_interval
# --------------------------------------------------------------------------- #


def _pairs(residual_mags):
    """Calibration pairs whose judge−reference residuals have the given mags."""
    return [(0.5 + m, 0.5) for m in residual_mags]


def test_interval_is_symmetric_before_clamping():
    pairs = _pairs([0.05, 0.1, 0.15, 0.2])  # q at alpha=0.5 is a middle residual
    iv = calibrate_interval(0.5, pairs, alpha=0.5)
    assert iv.lower == pytest.approx(0.5 - iv.half_width)
    assert iv.upper == pytest.approx(0.5 + iv.half_width)
    assert iv.midpoint == pytest.approx(0.5)


def test_interval_clamped_to_score_range():
    pairs = _pairs([0.3, 0.3, 0.3, 0.3])
    iv = calibrate_interval(0.95, pairs, alpha=0.5)
    assert iv.lower >= 0.0
    assert iv.upper <= 1.0
    # Near the top boundary the clamp pulls the midpoint back inside the range.
    assert iv.midpoint < 0.95


def test_width_shrinks_as_calibration_residuals_shrink():
    wide = calibrate_interval(0.5, _pairs([0.2, 0.3, 0.4, 0.5]), alpha=0.5)
    tight = calibrate_interval(0.5, _pairs([0.01, 0.02, 0.03, 0.04]), alpha=0.5)
    assert tight.width() < wide.width()


def test_single_calibration_pair_yields_full_range():
    iv = calibrate_interval(0.5, [(0.6, 0.5)], alpha=0.1)
    assert iv.lower == 0.0
    assert iv.upper == 1.0
    assert iv.half_width == pytest.approx(1.0)  # capped at range width


def test_interval_rejects_empty_calibration():
    with pytest.raises(ValueError):
        calibrate_interval(0.5, [], alpha=0.1)


def test_interval_rejects_degenerate_range():
    with pytest.raises(ValueError):
        calibrate_interval(0.5, [(0.6, 0.5)], score_range=(1.0, 1.0))


def test_interval_dataclass_helpers():
    iv = ConformalInterval(0.3, 0.7, 0.5, 0.9, 0.2, 10)
    assert iv.width() == pytest.approx(0.4)
    assert iv.contains(0.5) is True
    assert iv.contains(0.9) is False
    d = iv.to_dict()
    assert set(d) == {
        "lower",
        "upper",
        "midpoint",
        "coverage_level",
        "half_width",
        "n_calibration",
    }


# --------------------------------------------------------------------------- #
# Coverage guarantee (the whole point of conformal prediction)
# --------------------------------------------------------------------------- #


def test_empirical_coverage_meets_guarantee():
    """On exchangeable synthetic data, ~90% of held-out references fall within
    the 90%-level interval — the split-conformal marginal guarantee."""
    rng = random.Random(7)
    # judge = reference + symmetric noise; both exchangeable draws.
    data = [
        (r := rng.uniform(0.2, 0.8), r + rng.uniform(-0.1, 0.1)) for _ in range(400)
    ]
    cal, test = data[:200], data[200:]
    # Calibration pairs are (judge, reference); predict at the judge score.
    cal_pairs = [(judge, ref) for ref, judge in cal]
    covered = 0
    for ref, judge in test:
        iv = calibrate_interval(judge, cal_pairs, alpha=0.1)
        covered += iv.contains(ref)
    coverage = covered / len(test)
    assert coverage >= 0.85  # ≥ (1 − alpha) − finite-sample slack


# --------------------------------------------------------------------------- #
# Ensemble bridge
# --------------------------------------------------------------------------- #


def test_conformal_from_ensemble_wraps_median_aggregate():
    judges = [
        JudgeConfig(name="j1", seed=1, noise=0.05),
        JudgeConfig(name="j2", seed=2, noise=0.05),
        JudgeConfig(name="j3", seed=3, noise=0.05),
    ]
    result = ensemble_evaluate("a prompt", "a response", judges)
    cal = [(0.6, 0.55), (0.7, 0.66), (0.5, 0.52), (0.65, 0.6)]
    iv = conformal_from_ensemble(result, cal, alpha=0.5)
    assert iv.contains(result.median_aggregate)
    assert iv.coverage_level == pytest.approx(0.5)
    assert iv.n_calibration == 4
