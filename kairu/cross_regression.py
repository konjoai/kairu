"""Cross-model regression testing for kairu.

Compares per-criterion scores between two named models using data from a
LeaderboardStore and flags regressions exceeding a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from kairu.leaderboard import LeaderboardStore

__all__ = [
    "CriterionDelta",
    "CrossModelReport",
    "compare_models",
    "DEFAULT_REGRESSION_THRESHOLD",
]

DEFAULT_REGRESSION_THRESHOLD: float = 0.02


@dataclass(frozen=True)
class CriterionDelta:
    """Per-criterion score difference between two models."""

    criterion: str
    score_a: float
    score_b: float
    delta: float
    regressed: bool


@dataclass(frozen=True)
class CrossModelReport:
    """Regression report comparing two named models."""

    model_a: str
    model_b: str
    threshold: float
    days: int
    regressions: tuple[CriterionDelta, ...]
    improvements: tuple[CriterionDelta, ...]
    neutral: tuple[CriterionDelta, ...]
    aggregate_delta: float
    has_regressions: bool
    n_compared: int


def compare_models(
    model_a: str,
    model_b: str,
    store: LeaderboardStore,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
    days: int = 30,
) -> CrossModelReport:
    """Compare aggregate scores between two named models.

    Queries the leaderboard store for models ``model_a`` and ``model_b`` over
    the specified period and returns a :class:`CrossModelReport` that
    classifies the aggregate delta as a regression, improvement, or neutral.

    Parameters
    ----------
    model_a:
        Name of the baseline model.
    model_b:
        Name of the candidate model to compare against the baseline.
    store:
        A :class:`~kairu.leaderboard.LeaderboardStore` instance.
    threshold:
        Minimum absolute delta to consider non-neutral.
    days:
        Ranking window in days passed to
        :meth:`~kairu.leaderboard.LeaderboardStore.rank`.

    Returns
    -------
    CrossModelReport

    Raises
    ------
    ValueError
        When either model is not found in the leaderboard.
    """
    rows = store.rank(days=days, limit=200)

    row_a = next((r for r in rows if r.model == model_a), None)
    if row_a is None:
        raise ValueError(
            f"model '{model_a}' not found in the leaderboard for the last {days} days"
        )

    row_b = next((r for r in rows if r.model == model_b), None)
    if row_b is None:
        raise ValueError(
            f"model '{model_b}' not found in the leaderboard for the last {days} days"
        )

    score_a = row_a.mean_score
    score_b = row_b.mean_score
    delta = score_b - score_a

    regressed = delta < -threshold
    cd = CriterionDelta(
        criterion="aggregate",
        score_a=score_a,
        score_b=score_b,
        delta=delta,
        regressed=regressed,
    )

    if delta < -threshold:
        regressions: tuple[CriterionDelta, ...] = (cd,)
        improvements: tuple[CriterionDelta, ...] = ()
        neutral: tuple[CriterionDelta, ...] = ()
    elif delta > threshold:
        regressions = ()
        improvements = (cd,)
        neutral = ()
    else:
        regressions = ()
        improvements = ()
        neutral = (cd,)

    return CrossModelReport(
        model_a=model_a,
        model_b=model_b,
        threshold=threshold,
        days=days,
        regressions=regressions,
        improvements=improvements,
        neutral=neutral,
        aggregate_delta=delta,
        has_regressions=len(regressions) > 0,
        n_compared=1,
    )
