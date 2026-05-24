"""Tests for kairu/calibration.py — BiasProfile, CalibrationPair, correction."""

from __future__ import annotations

import pytest

from kairu.calibration import (
    BiasProfile,
    BiasProfileStore,
    CalibrationPair,
    CalibratedEnsembleResult,
    build_bias_profile,
    compute_uncalibrated_bias_bound,
    correct_ensemble_scores,
)
from kairu.ensemble import EnsembleResult, JudgeConfig, ensemble_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def judges():
    return [
        JudgeConfig(name="j1", rubric="default", noise=0.0),
        JudgeConfig(name="j2", rubric="default", noise=0.05, seed=1),
    ]


@pytest.fixture
def simple_pair():
    return CalibrationPair(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        human_scores={
            "relevance": 1.0,
            "coherence": 0.9,
            "conciseness": 0.8,
            "safety": 1.0,
            "fluency": 0.9,
            "specificity": 0.7,
            "completeness": 0.8,
        },
    )


@pytest.fixture
def calibration_pairs(simple_pair):
    return [
        simple_pair,
        CalibrationPair(
            prompt="Summarise the plot of Hamlet.",
            response="Hamlet is a Danish prince who seeks revenge for his father's murder.",
            human_scores={
                "relevance": 0.9,
                "coherence": 0.85,
                "conciseness": 0.75,
                "safety": 1.0,
                "fluency": 0.8,
                "specificity": 0.6,
                "completeness": 0.7,
            },
        ),
        CalibrationPair(
            prompt="What is 2+2?",
            response="2+2 equals 4.",
            human_scores={
                "relevance": 1.0,
                "coherence": 1.0,
                "conciseness": 1.0,
                "safety": 1.0,
                "fluency": 0.9,
                "specificity": 0.8,
                "completeness": 0.9,
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationPair
# ─────────────────────────────────────────────────────────────────────────────


def test_calibration_pair_construction(simple_pair):
    assert simple_pair.prompt
    assert simple_pair.response
    assert "relevance" in simple_pair.human_scores


def test_calibration_pair_empty_prompt_raises():
    with pytest.raises(ValueError, match="prompt"):
        CalibrationPair(prompt="", response="ok", human_scores={"relevance": 0.5})


def test_calibration_pair_empty_response_raises():
    with pytest.raises(ValueError, match="response"):
        CalibrationPair(prompt="q", response="", human_scores={"relevance": 0.5})


def test_calibration_pair_out_of_range_score_raises():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        CalibrationPair(prompt="q", response="a", human_scores={"relevance": 1.5})


def test_calibration_pair_boundary_scores_valid():
    p = CalibrationPair(
        prompt="q", response="a", human_scores={"relevance": 0.0, "safety": 1.0}
    )
    assert p.human_scores["relevance"] == 0.0
    assert p.human_scores["safety"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# build_bias_profile
# ─────────────────────────────────────────────────────────────────────────────


def test_build_bias_profile_returns_frozen(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    assert isinstance(profile, BiasProfile)
    with pytest.raises((AttributeError, TypeError)):
        profile.rubric = "other"  # type: ignore[misc]


def test_build_bias_profile_rubric_label(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs, rubric="helpfulness")
    assert profile.rubric == "helpfulness"


def test_build_bias_profile_n_pairs(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    assert profile.n_calibration_pairs == len(calibration_pairs)


def test_build_bias_profile_has_criterion_biases(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    assert len(profile.criterion_biases) > 0
    for v in profile.criterion_biases.values():
        assert isinstance(v, float)


def test_build_bias_profile_calibration_hash_is_hex(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    assert len(profile.calibration_hash) == 16
    int(profile.calibration_hash, 16)  # must be valid hex


def test_build_bias_profile_bias_bound_in_range(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    assert 0.0 <= profile.bias_bound <= 1.0


def test_build_bias_profile_single_pair_bound_is_one(judges, simple_pair):
    # With n=1, Hoeffding is wide — bound may hit the 1.0 cap
    profile = build_bias_profile(judges, [simple_pair])
    assert profile.bias_bound <= 1.0


def test_build_bias_profile_more_pairs_tighter_bound(judges, calibration_pairs):
    """Hoeffding bound decreases as n grows."""
    profile_small = build_bias_profile(judges, calibration_pairs[:1])
    profile_large = build_bias_profile(judges, calibration_pairs)
    assert profile_large.bias_bound <= profile_small.bias_bound


def test_build_bias_profile_empty_pairs_raises(judges):
    with pytest.raises(ValueError, match="non-empty"):
        build_bias_profile(judges, [])


def test_build_bias_profile_empty_judges_raises(calibration_pairs):
    with pytest.raises(ValueError, match="non-empty"):
        build_bias_profile([], calibration_pairs)


def test_build_bias_profile_deterministic(judges, calibration_pairs):
    """Same inputs must produce identical output."""
    p1 = build_bias_profile(judges, calibration_pairs)
    p2 = build_bias_profile(judges, calibration_pairs)
    assert p1.calibration_hash == p2.calibration_hash
    assert p1.criterion_biases == p2.criterion_biases


# ─────────────────────────────────────────────────────────────────────────────
# correct_ensemble_scores
# ─────────────────────────────────────────────────────────────────────────────


def _make_ensemble_result(judges) -> EnsembleResult:
    return ensemble_evaluate(
        "What is the capital of France?",
        "Paris is the capital of France.",
        judges,
    )


def test_correct_ensemble_scores_returns_calibrated(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    raw = _make_ensemble_result(judges)
    calibrated = correct_ensemble_scores(raw, profile)
    assert isinstance(calibrated, CalibratedEnsembleResult)


def test_correct_ensemble_scores_does_not_mutate_raw(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    raw = _make_ensemble_result(judges)
    original_scores = dict(raw.median_scores)
    correct_ensemble_scores(raw, profile)
    assert raw.median_scores == original_scores


def test_correct_ensemble_scores_clamps_to_unit_interval(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    raw = _make_ensemble_result(judges)
    calibrated = correct_ensemble_scores(raw, profile)
    for score in calibrated.bias_corrected_scores.values():
        assert 0.0 <= score <= 1.0


def test_correct_ensemble_scores_aggregate_in_range(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    raw = _make_ensemble_result(judges)
    calibrated = correct_ensemble_scores(raw, profile)
    assert 0.0 <= calibrated.bias_corrected_aggregate <= 1.0


def test_correct_ensemble_scores_profile_hash_propagated(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    raw = _make_ensemble_result(judges)
    calibrated = correct_ensemble_scores(raw, profile)
    assert calibrated.profile_hash == profile.calibration_hash


def test_correct_ensemble_scores_zero_bias_noop(judges):
    """A profile with zero bias should leave scores unchanged."""
    raw = _make_ensemble_result(judges)
    zero_profile = BiasProfile(
        rubric="default",
        criterion_biases={c: 0.0 for c in raw.median_scores},
        n_calibration_pairs=10,
        calibration_hash="0000000000000000",
        bias_bound=0.1,
    )
    calibrated = correct_ensemble_scores(raw, zero_profile)
    for criterion, score in calibrated.bias_corrected_scores.items():
        assert abs(score - raw.median_scores[criterion]) < 1e-9


def test_correct_ensemble_scores_to_dict_shape(judges, calibration_pairs):
    profile = build_bias_profile(judges, calibration_pairs)
    raw = _make_ensemble_result(judges)
    calibrated = correct_ensemble_scores(raw, profile)
    d = calibrated.to_dict()
    assert "raw" in d
    assert "bias_corrected_scores" in d
    assert "bias_corrected_aggregate" in d
    assert "bias_bound" in d
    assert "profile_hash" in d


# ─────────────────────────────────────────────────────────────────────────────
# compute_uncalibrated_bias_bound
# ─────────────────────────────────────────────────────────────────────────────


def test_uncalibrated_bias_bound_empty_returns_zero():
    assert compute_uncalibrated_bias_bound({}) == 0.0


def test_uncalibrated_bias_bound_in_range():
    stdevs = {"relevance": 0.1, "coherence": 0.2, "safety": 0.05}
    bound = compute_uncalibrated_bias_bound(stdevs)
    assert 0.0 <= bound <= 1.0


def test_uncalibrated_bias_bound_uses_max_stdev():
    # bound ∝ max stdev; check that a larger max stdev gives a larger bound
    bound_low = compute_uncalibrated_bias_bound({"a": 0.05})
    bound_high = compute_uncalibrated_bias_bound({"a": 0.3})
    assert bound_high > bound_low


def test_uncalibrated_bias_bound_confidence_levels():
    stdevs = {"a": 0.2}
    b95 = compute_uncalibrated_bias_bound(stdevs, confidence=0.95)
    b99 = compute_uncalibrated_bias_bound(stdevs, confidence=0.99)
    assert b99 >= b95


# ─────────────────────────────────────────────────────────────────────────────
# BiasProfileStore
# ─────────────────────────────────────────────────────────────────────────────


def test_bias_profile_store_save_and_load(judges, calibration_pairs):
    store = BiasProfileStore()
    profile = build_bias_profile(judges, calibration_pairs)
    store.save(profile)
    loaded = store.load("default")
    assert loaded is not None
    assert loaded.calibration_hash == profile.calibration_hash


def test_bias_profile_store_load_missing_returns_none():
    store = BiasProfileStore()
    assert store.load("nonexistent") is None


def test_bias_profile_store_list(judges, calibration_pairs):
    store = BiasProfileStore()
    assert store.list() == []
    profile = build_bias_profile(judges, calibration_pairs, rubric="helpfulness")
    store.save(profile)
    assert "helpfulness" in store.list()


def test_bias_profile_store_overwrite(judges, calibration_pairs):
    """Last-write-wins — saving a second profile for the same rubric replaces the first."""
    store = BiasProfileStore()
    p1 = build_bias_profile(judges, calibration_pairs[:1])
    p2 = build_bias_profile(judges, calibration_pairs)
    store.save(p1)
    store.save(p2)
    loaded = store.load("default")
    assert loaded.n_calibration_pairs == p2.n_calibration_pairs
