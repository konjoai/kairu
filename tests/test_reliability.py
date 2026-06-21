"""Tests for kairu.reliability — Cronbach's alpha, ICC(2,1), Fleiss' kappa.

Expected values are hand-derived from the textbook formulas so the tests
pin the arithmetic, not just its self-consistency.
"""

from __future__ import annotations

import pytest

from kairu.ensemble import JudgeConfig, ensemble_evaluate
from kairu.reliability import (
    ReliabilityReport,
    compute_reliability,
    cronbach_alpha,
    fleiss_kappa,
    intraclass_correlation,
    reliability_from_ensemble,
)

# Judges × criteria, perfectly covarying columns → Cronbach alpha = 1.0.
PERFECT_CONSISTENCY = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
]


# ── Cronbach's alpha ───────────────────────────────────────────────────────


def test_cronbach_perfect_consistency():
    assert cronbach_alpha(PERFECT_CONSISTENCY) == pytest.approx(1.0)


def test_cronbach_needs_two_judges():
    assert cronbach_alpha([[0.1, 0.2, 0.3]]) is None


def test_cronbach_needs_two_criteria():
    assert cronbach_alpha([[0.1], [0.2], [0.3]]) is None


def test_cronbach_zero_total_variance_is_none():
    # Identical judges → no total-score variance → alpha undefined.
    assert cronbach_alpha([[0.2, 0.8], [0.2, 0.8]]) is None


def test_cronbach_in_range_for_noisy_matrix():
    # Row sums differ (1.2 / 1.1 / 1.6) so total variance is non-zero.
    alpha = cronbach_alpha([[0.1, 0.2, 0.9], [0.8, 0.1, 0.2], [0.5, 0.6, 0.5]])
    assert alpha is not None
    assert alpha <= 1.0


# ── ICC(2,1) ───────────────────────────────────────────────────────────────


def test_icc_perfect_agreement():
    # criteria × judges; raters agree exactly, subjects differ → ICC = 1.0.
    assert intraclass_correlation([[0.2, 0.2], [0.8, 0.8]]) == pytest.approx(1.0)


def test_icc_needs_two_subjects():
    assert intraclass_correlation([[0.2, 0.2]]) is None


def test_icc_needs_two_raters():
    assert intraclass_correlation([[0.2], [0.8]]) is None


def test_icc_low_for_disagreement():
    # A large systematic between-rater offset depresses absolute-agreement ICC.
    icc = intraclass_correlation([[0.2, 0.6], [0.3, 0.7]])
    assert icc is not None
    assert icc < 0.5


# ── Fleiss' kappa ──────────────────────────────────────────────────────────


def test_fleiss_perfect_agreement_balanced():
    # crit0 both fail, crit1 both pass → perfect agreement, balanced → kappa 1.
    assert fleiss_kappa([[0.2, 0.2], [0.8, 0.8]]) == pytest.approx(1.0)


def test_fleiss_total_disagreement():
    # Every criterion split 1/1 → kappa = -1.0.
    assert fleiss_kappa([[0.9, 0.1], [0.1, 0.9]]) == pytest.approx(-1.0)


def test_fleiss_no_category_variation_is_none():
    # All scores pass → P_e == 1 → kappa undefined.
    assert fleiss_kappa([[0.8, 0.9], [0.7, 0.8]]) is None


def test_fleiss_respects_threshold():
    # At threshold 0.5 both pass on crit0; at 0.95 both fail — still agreement,
    # but flips the category, exercising the threshold path.
    assert fleiss_kappa([[0.6, 0.6], [0.2, 0.2]], threshold=0.5) == pytest.approx(1.0)


def test_fleiss_needs_two_criteria():
    assert fleiss_kappa([[0.2, 0.8]]) is None


# ── compute_reliability ────────────────────────────────────────────────────


def test_compute_reliability_transposes_for_icc_and_kappa():
    # judges × criteria: 2 judges agree, 2 criteria differ.
    report = compute_reliability([[0.2, 0.8], [0.2, 0.8]])
    assert isinstance(report, ReliabilityReport)
    assert report.cronbach_alpha is None  # identical judges
    assert report.icc == pytest.approx(1.0)
    assert report.fleiss_kappa == pytest.approx(1.0)
    assert report.n_judges == 2
    assert report.n_criteria == 2


def test_compute_reliability_labels():
    report = compute_reliability(PERFECT_CONSISTENCY)
    assert report.cronbach_label == "excellent"
    assert report.icc_label in {"good", "excellent", "moderate", "poor"}


def test_compute_reliability_undefined_labels():
    report = compute_reliability([[0.5]])
    assert report.cronbach_alpha is None
    assert report.cronbach_label == "undefined"
    assert report.icc_label == "undefined"
    assert report.fleiss_label == "undefined"


def test_compute_reliability_empty_matrix():
    report = compute_reliability([[]])
    assert report.n_criteria == 0
    assert report.cronbach_alpha is None


def test_reliability_to_dict_round_trips():
    d = compute_reliability(PERFECT_CONSISTENCY).to_dict()
    assert set(d) == {
        "cronbach_alpha",
        "cronbach_label",
        "icc",
        "icc_label",
        "fleiss_kappa",
        "fleiss_label",
        "pass_threshold",
        "n_judges",
        "n_criteria",
    }


# ── reliability_from_ensemble ──────────────────────────────────────────────


def test_reliability_from_ensemble_shape():
    judges = [
        JudgeConfig(name="a", noise=0.05, seed=1),
        JudgeConfig(name="b", noise=0.05, seed=2),
        JudgeConfig(name="c", noise=0.05, seed=3),
    ]
    result = ensemble_evaluate("Explain gravity.", "Gravity attracts mass.", judges)
    report = reliability_from_ensemble(result)
    assert report.n_judges == 3
    assert report.n_criteria >= 2


def test_reliability_from_ensemble_single_judge_metrics_none():
    result = ensemble_evaluate("Q?", "An answer.", [JudgeConfig(name="solo")])
    report = reliability_from_ensemble(result)
    assert report.n_judges == 1
    assert report.cronbach_alpha is None
    assert report.icc is None
    assert report.fleiss_kappa is None
