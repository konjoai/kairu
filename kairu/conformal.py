"""Split conformal prediction intervals for LLM-judge scores.

The reliability module (:mod:`kairu.reliability`) ships Cronbach's alpha,
ICC(2,1) and Fleiss' kappa — all *retrospective, distribution-parametric*
estimators that describe a judges × criteria matrix after the fact. None of
them answers the deployment question a practitioner actually asks: *given this
judge's score, what is a defensible interval around the true quality, with a
coverage guarantee that holds regardless of the judge's score distribution?*

Split conformal prediction (Vovk; applied to LLM judges by Sheng et al.,
EMNLP 2025, arXiv:2509.18658) answers exactly that. From a calibration set of
``(judge_score, reference_score)`` pairs it computes the nonconformity scores
(absolute residuals), takes their finite-sample-corrected ``(1 - alpha)``
quantile ``q``, and emits the symmetric interval ``[score - q, score + q]``.
For any *exchangeable* new pair the interval covers the reference score with
probability at least ``1 - alpha`` — **distribution-free**, no assumption on
the judge's score distribution.

Two judge-specific refinements from the paper:

* **Ordinal boundary adjustment** — the interval is clamped to the valid score
  range (``[0, 1]`` by default), so it never claims impossible scores.
* **Midpoint point-estimate** — the midpoint of the *clamped* interval is a
  lower-bias estimate than the raw score near a boundary, where clamping pulls
  the interval (and thus its midpoint) back inside the range.

Pure stdlib + :mod:`kairu.ensemble` — no ML deps. The quantile follows kairu's
existing sorted-list / nearest-rank pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from kairu.ensemble import EnsembleResult

# 90 % coverage by default (alpha = 0.1).
DEFAULT_ALPHA: float = 0.1
# kairu scores live in [0, 1]; the interval is clamped to this range.
DEFAULT_SCORE_RANGE: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class ConformalInterval:
    """A distribution-free prediction interval around a judge score.

    Attributes
    ----------
    lower, upper:
        Interval endpoints, clamped to the score range.
    midpoint:
        Midpoint of the clamped interval — the lower-bias point estimate.
    coverage_level:
        The guaranteed marginal coverage ``1 - alpha``.
    half_width:
        The conformal quantile ``q`` actually applied (capped at the score-range
        width when the calibration set is too small for a finite bound).
    n_calibration:
        Number of calibration pairs the quantile was computed from.
    """

    lower: float
    upper: float
    midpoint: float
    coverage_level: float
    half_width: float
    n_calibration: int

    def width(self) -> float:
        """Total width of the (clamped) interval."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Whether ``value`` falls within the closed interval."""
        return self.lower <= value <= self.upper

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the interval."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "midpoint": self.midpoint,
            "coverage_level": self.coverage_level,
            "half_width": self.half_width,
            "n_calibration": self.n_calibration,
        }


def conformal_quantile(
    residuals: Sequence[float], alpha: float = DEFAULT_ALPHA
) -> float:
    """Finite-sample split-conformal ``(1 - alpha)`` quantile of ``|residuals|``.

    Uses the conformal rank ``ceil((n + 1) * (1 - alpha))``. When that rank
    exceeds ``n`` the calibration set is too small to certify a finite bound at
    this ``alpha``, and the honest answer is ``+inf`` (an interval that must
    span the whole range); callers clamp it to their score range.

    Parameters
    ----------
    residuals:
        Calibration residuals; their absolute values are used.
    alpha:
        Miscoverage level in ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = len(residuals)
    if n == 0:
        raise ValueError("at least one calibration residual is required")
    ordered = sorted(abs(r) for r in residuals)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return ordered[rank - 1]


def calibrate_interval(
    prediction: float,
    calibration_pairs: Sequence[Tuple[float, float]],
    *,
    alpha: float = DEFAULT_ALPHA,
    score_range: Tuple[float, float] = DEFAULT_SCORE_RANGE,
) -> ConformalInterval:
    """Build a split-conformal interval around ``prediction``.

    Parameters
    ----------
    prediction:
        The judge score to wrap in an interval.
    calibration_pairs:
        Historical ``(judge_score, reference_score)`` pairs; the residual of
        each is ``judge_score - reference_score``.
    alpha:
        Miscoverage level — the interval guarantees ``1 - alpha`` coverage.
    score_range:
        ``(low, high)`` bounds the interval is clamped to (ordinal adjustment).
    """
    if not calibration_pairs:
        raise ValueError("at least one calibration pair is required")
    low, high = score_range
    if high <= low:
        raise ValueError("score_range must have high > low")

    residuals = [judge - reference for judge, reference in calibration_pairs]
    q = conformal_quantile(residuals, alpha)
    # Cap at the range width so an unbounded quantile becomes the full range and
    # the stored half_width stays finite / JSON-serialisable.
    half_width = min(q, high - low)

    lower = max(low, prediction - half_width)
    upper = min(high, prediction + half_width)
    return ConformalInterval(
        lower=lower,
        upper=upper,
        midpoint=(lower + upper) / 2.0,
        coverage_level=1.0 - alpha,
        half_width=half_width,
        n_calibration=len(calibration_pairs),
    )


def conformal_from_ensemble(
    result: EnsembleResult,
    calibration_pairs: Sequence[Tuple[float, float]],
    *,
    alpha: float = DEFAULT_ALPHA,
    score_range: Tuple[float, float] = DEFAULT_SCORE_RANGE,
) -> ConformalInterval:
    """Wrap an ensemble's median aggregate in a conformal interval.

    Uses ``result.median_aggregate`` as the prediction; ``calibration_pairs``
    are prior ``(ensemble_aggregate, reference_score)`` observations.

    Parameters
    ----------
    result:
        The ensemble verdict to bound.
    calibration_pairs:
        Historical ``(aggregate, reference)`` pairs.
    alpha, score_range:
        As in :func:`calibrate_interval`.
    """
    return calibrate_interval(
        result.median_aggregate,
        calibration_pairs,
        alpha=alpha,
        score_range=score_range,
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SCORE_RANGE",
    "ConformalInterval",
    "conformal_quantile",
    "calibrate_interval",
    "conformal_from_ensemble",
]
