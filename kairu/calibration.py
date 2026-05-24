"""Judge bias calibration — bias profile estimation and A-BB-style bias bounds.

Two research results motivate this module:

1. **Agreeableness bias** (arXiv:2510.11822 — "Beyond Consensus"):
   LLM judges exhibit a systematic true-positive rate / true-negative rate
   asymmetry (TPR ~96%, TNR <25%).  Even deterministic heuristic scorers
   can drift from human labels in a consistent direction.  A small set of
   human-annotated calibration pairs exposes the per-criterion mean offset,
   which can be subtracted from ensemble scores — halving max absolute error
   relative to majority-vote baselines.

2. **Bias-Bounded Evaluation** (arXiv:2603.05485 — "A-BB"):
   Using Hoeffding's inequality over bounded [0, 1] scores, we derive a
   95%-confidence upper bound on the maximum absolute bias on an unseen item
   given *n* calibration pairs.  This gives a formal PAC-style guarantee
   without requiring a specific model family.

Architecture
------------
*  ``CalibrationPair`` — one human-annotated (prompt, response, scores) example.
*  ``BiasProfile`` — per-criterion additive bias + Hoeffding bias bound,
   computed from N calibration pairs.
*  ``CalibratedEnsembleResult`` — wraps an ``EnsembleResult`` with
   ``bias_corrected_scores`` and the profile's ``bias_bound``.
*  ``build_bias_profile(judges, pairs, ...)`` — runs the ensemble on each
   calibration pair, computes mean(judge_score - human_score) per criterion,
   and derives the Hoeffding bound.
*  ``correct_ensemble_scores(result, profile)`` — subtracts per-criterion
   bias and clamps to [0, 1].
*  ``compute_uncalibrated_bias_bound(stdev_scores)`` — when no calibration
   data is available, the inter-judge standard deviation is a conservative
   proxy for potential bias (a judge cannot be biased beyond its own spread).

No ML deps — pure stdlib + kairu.ensemble.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from kairu.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    EnsembleResult,
    JudgeConfig,
    ensemble_evaluate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationPair:
    """One human-annotated example for bias estimation.

    Parameters
    ----------
    prompt:
        The input prompt shown to the model.
    response:
        The model response being evaluated.
    human_scores:
        Per-criterion human judgments in [0, 1].  Missing criteria are
        excluded from bias estimation for that criterion.
    """

    prompt: str
    response: str
    human_scores: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("CalibrationPair.prompt must be non-empty")
        if not self.response:
            raise ValueError("CalibrationPair.response must be non-empty")
        bad = {k: v for k, v in self.human_scores.items() if not (0.0 <= v <= 1.0)}
        if bad:
            raise ValueError(f"human_scores must be in [0, 1]; got {bad}")


@dataclass(frozen=True)
class BiasProfile:
    """Per-criterion additive bias estimates from calibration pairs.

    ``criterion_biases[c] = mean(judge_score[c] - human_score[c])`` over
    all calibration pairs that include criterion *c*.  A positive value
    means the ensemble *over-rates* that criterion relative to human labels;
    subtract to correct.

    ``bias_bound`` is a Hoeffding 95%-confidence upper bound on the maximum
    absolute bias on a new unseen item:

        bias_bound = max|criterion_biases| + sqrt(log(40) / (2n))

    (log(2/0.05) = log(40), Hoeffding for [0,1]-bounded differences.)
    """

    rubric: str
    criterion_biases: Dict[str, float]
    n_calibration_pairs: int
    calibration_hash: str
    bias_bound: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric": self.rubric,
            "criterion_biases": dict(self.criterion_biases),
            "n_calibration_pairs": self.n_calibration_pairs,
            "calibration_hash": self.calibration_hash,
            "bias_bound": self.bias_bound,
        }


@dataclass(frozen=True)
class CalibratedEnsembleResult:
    """``EnsembleResult`` with per-criterion bias correction applied.

    ``bias_corrected_scores[c] = clamp(median_scores[c] - bias[c], 0, 1)``.

    ``bias_corrected_aggregate`` is the unweighted mean of corrected scores
    — an approximation when the rubric uses non-uniform weights, but
    consistent and reproducible regardless of which rubric was used.

    ``bias_bound`` propagates directly from the ``BiasProfile``.
    """

    raw: EnsembleResult
    bias_corrected_scores: Dict[str, float]
    bias_corrected_aggregate: float
    bias_bound: float
    profile_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw.to_dict(),
            "bias_corrected_scores": dict(self.bias_corrected_scores),
            "bias_corrected_aggregate": self.bias_corrected_aggregate,
            "bias_bound": self.bias_bound,
            "profile_hash": self.profile_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────


def build_bias_profile(
    judges: Sequence[JudgeConfig],
    calibration_pairs: Sequence[CalibrationPair],
    *,
    rubric: str = "default",
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> BiasProfile:
    """Estimate per-criterion bias from human-annotated calibration pairs.

    For each pair the ensemble is run and each criterion's judge score is
    compared to the human label.  The mean signed difference is the bias.

    Parameters
    ----------
    judges:
        Non-empty sequence of judge configurations — same as
        :func:`kairu.ensemble.ensemble_evaluate`.
    calibration_pairs:
        At least one ``CalibrationPair`` with human labels.  The more
        pairs, the tighter the Hoeffding bound.
    rubric:
        Rubric name recorded in the returned ``BiasProfile``.
    disagreement_threshold:
        Passed through to :func:`kairu.ensemble.ensemble_evaluate`.

    Returns
    -------
    BiasProfile
        Frozen dataclass with per-criterion biases and the 95%-confidence
        Hoeffding bias bound.
    """
    if not judges:
        raise ValueError("judges must be non-empty")
    if not calibration_pairs:
        raise ValueError("calibration_pairs must be non-empty")

    criterion_sum: Dict[str, float] = {}
    criterion_count: Dict[str, int] = {}
    hasher = hashlib.sha256()

    for pair in calibration_pairs:
        hasher.update(pair.prompt.encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(pair.response.encode("utf-8"))

        result = ensemble_evaluate(
            pair.prompt,
            pair.response,
            judges,
            disagreement_threshold=disagreement_threshold,
        )
        for criterion, judge_score in result.median_scores.items():
            if criterion not in pair.human_scores:
                continue
            diff = judge_score - pair.human_scores[criterion]
            criterion_sum[criterion] = criterion_sum.get(criterion, 0.0) + diff
            criterion_count[criterion] = criterion_count.get(criterion, 0) + 1

    criterion_biases: Dict[str, float] = {
        c: criterion_sum[c] / criterion_count[c] for c in criterion_count
    }

    n = len(calibration_pairs)
    # Hoeffding: at 95% confidence, P(|mean_bias - true_bias| > t) <= 0.05
    # → t = sqrt(log(2/delta) / (2n)) = sqrt(log(40) / (2n))
    hoeffding_t = math.sqrt(math.log(40.0) / (2.0 * n)) if n >= 1 else 1.0
    max_abs_bias = (
        max(abs(b) for b in criterion_biases.values()) if criterion_biases else 0.0
    )
    bias_bound = min(1.0, max_abs_bias + hoeffding_t)

    return BiasProfile(
        rubric=rubric,
        criterion_biases=criterion_biases,
        n_calibration_pairs=n,
        calibration_hash=hasher.hexdigest()[:16],
        bias_bound=bias_bound,
    )


def correct_ensemble_scores(
    result: EnsembleResult,
    profile: BiasProfile,
) -> CalibratedEnsembleResult:
    """Apply a ``BiasProfile`` correction to an ``EnsembleResult``.

    Each criterion score is shifted by ``-bias[c]`` and clamped to [0, 1].
    Criteria not covered by the profile are passed through unchanged.
    The corrected aggregate is the unweighted mean of corrected scores.
    """
    corrected: Dict[str, float] = {}
    for criterion, score in result.median_scores.items():
        bias = profile.criterion_biases.get(criterion, 0.0)
        corrected[criterion] = max(0.0, min(1.0, score - bias))

    agg = sum(corrected.values()) / len(corrected) if corrected else 0.0

    return CalibratedEnsembleResult(
        raw=result,
        bias_corrected_scores=corrected,
        bias_corrected_aggregate=agg,
        bias_bound=profile.bias_bound,
        profile_hash=profile.calibration_hash,
    )


def compute_uncalibrated_bias_bound(
    stdev_scores: Dict[str, float],
    *,
    confidence: float = 0.95,
) -> float:
    """Conservative bias bound from inter-judge disagreement (no human labels).

    A judge cannot exhibit bias beyond its own inter-judge spread, so
    ``max(stdev_scores) * z`` is a conservative upper bound where *z* is
    the normal-distribution quantile for the requested confidence level.

    This is the fallback for ``POST /evaluate/ensemble`` when no calibration
    set is provided — it surfaces in the response as ``uncalibrated_bias_bound``
    without requiring any additional data.
    """
    if not stdev_scores:
        return 0.0
    max_stdev = max(stdev_scores.values())
    if confidence >= 0.99:
        z = 2.576
    elif confidence >= 0.95:
        z = 1.960
    else:
        z = 1.645
    return min(1.0, max_stdev * z)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory profile store (mirrors BaselineStore pattern)
# ─────────────────────────────────────────────────────────────────────────────


class BiasProfileStore:
    """In-memory store for ``BiasProfile`` objects, keyed by rubric name.

    One store per ``create_app()`` instance.  Profiles are keyed by rubric
    name; storing a second profile for the same rubric overwrites the first
    (last-write-wins — callers re-calibrate as human labels accumulate).
    """

    def __init__(self) -> None:
        self._store: Dict[str, BiasProfile] = {}

    def save(self, profile: BiasProfile) -> None:
        self._store[profile.rubric] = profile

    def load(self, rubric: str) -> Optional[BiasProfile]:
        return self._store.get(rubric)

    def list(self) -> list[str]:
        return list(self._store.keys())


__all__ = [
    "BiasProfile",
    "BiasProfileStore",
    "CalibrationPair",
    "CalibratedEnsembleResult",
    "build_bias_profile",
    "correct_ensemble_scores",
    "compute_uncalibrated_bias_bound",
]
