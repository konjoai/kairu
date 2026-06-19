"""Round-robin judge allocation + coverage-correct batch intervals.

:mod:`kairu.ensemble` runs every judge on every item and median-aggregates —
right for one high-stakes item, but at batch scale it costs N·K calls and,
worse, only shrinks per-item outliers while leaving a *systematic* per-judge
offset fully intact in the aggregate. **CyclicJudge** (arXiv:2603.01865)
proves round-robin allocation — rotate which single judge scores each item —
balances every judge across the batch at single-judge cost, so systematic
offsets cancel in the batch mean instead of summing.

:func:`batch_mean_interval` then builds the Student-t interval over
*independent per-item aggregates*. That sampling unit matters: a CI built
over the *criteria within one item* (as :func:`kairu.significance.paired_t_test`
does) has ~0 % empirical coverage for "true mean quality" — Causal Judge
Evaluation, arXiv:2512.11150. The items are the independent draws.

:func:`variance_components` runs a one-observation-per-cell two-way ANOVA
over a judge × item grid, splitting total variance into a *judge* component
(systematic disagreement), an *item* component (real signal), and a residual.
``judge_variance_fraction`` is the judge share; when small, round-robin is safe.

No ML deps — pure stdlib + :mod:`kairu.ensemble`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from kairu.ensemble import JudgeConfig, JudgeScore, judge_evaluate

# Reused inside the package: the bisection t critical value already proven
# correct by tests/test_significance.py. Importing it keeps a single source
# of truth for the Student-t quantile rather than re-deriving it here.
from kairu.significance import _student_t_critical

DEFAULT_CONFIDENCE: float = 0.95


# ─────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MeanInterval:
    """Confidence interval on a batch mean aggregate.

    Attributes
    ----------
    mean:
        Point estimate — the unweighted mean of the per-item aggregates.
    lo, hi:
        Lower / upper bounds of the two-sided interval at ``confidence``.
    n:
        Number of independent items the interval is built from.
    stdev:
        Sample standard deviation of the per-item aggregates (``ddof=1``);
        ``0.0`` when ``n < 2``.
    confidence:
        Nominal coverage level, e.g. ``0.95``.
    """

    mean: float
    lo: float
    hi: float
    n: int
    stdev: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the interval."""
        return {
            "mean": self.mean,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "stdev": self.stdev,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class JudgeLoad:
    """One judge's share of a round-robin batch.

    Attributes
    ----------
    judge:
        Judge name (matches :class:`kairu.ensemble.JudgeConfig.name`).
    n_items:
        Number of items routed to this judge. ``0`` is possible when there
        are more judges than items.
    mean_aggregate:
        Mean aggregate over this judge's items; ``0.0`` when ``n_items == 0``.
    """

    judge: str
    n_items: int
    mean_aggregate: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the load."""
        return {
            "judge": self.judge,
            "n_items": self.n_items,
            "mean_aggregate": self.mean_aggregate,
        }


@dataclass(frozen=True)
class VarianceComponents:
    """Two-way ANOVA decomposition of a judge × item score grid.

    Sums of squares are split into a *judge* effect (systematic per-judge
    offset), an *item* effect (genuine quality signal), and a *residual*.
    ``judge_variance_fraction = ss_judge / ss_total`` is the share of total
    variance attributable to systematic judge disagreement — the quantity
    round-robin allocation cancels in the batch mean.

    Attributes
    ----------
    ss_judge, ss_item, ss_residual, ss_total:
        Sums of squares for each effect and the total.
    judge_variance_fraction, item_variance_fraction:
        ``ss_judge / ss_total`` and ``ss_item / ss_total``; both ``0.0``
        when ``ss_total == 0`` (every cell identical).
    n_judges, n_items:
        Grid dimensions.
    """

    ss_judge: float
    ss_item: float
    ss_residual: float
    ss_total: float
    judge_variance_fraction: float
    item_variance_fraction: float
    n_judges: int
    n_items: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the decomposition."""
        return {
            "ss_judge": self.ss_judge,
            "ss_item": self.ss_item,
            "ss_residual": self.ss_residual,
            "ss_total": self.ss_total,
            "judge_variance_fraction": self.judge_variance_fraction,
            "item_variance_fraction": self.item_variance_fraction,
            "n_judges": self.n_judges,
            "n_items": self.n_items,
        }


@dataclass(frozen=True)
class CyclicEvalReport:
    """Result of a round-robin batch evaluation.

    Attributes
    ----------
    assignments:
        ``(item_index, judge_name)`` for every item, in item order.
    per_item:
        The :class:`kairu.ensemble.JudgeScore` for each item, in item order.
    judge_loads:
        Per-judge load and mean — the balance diagnostic.
    mean_aggregate:
        Unweighted mean of every item's aggregate.
    balance:
        ``max(n_items) - min(n_items)`` across judges. Round-robin
        guarantees this is ``0`` or ``1``; anything larger signals a bug.
    interval:
        Coverage-correct Student-t interval on ``mean_aggregate``.
    """

    assignments: Tuple[Tuple[int, str], ...]
    per_item: Tuple[JudgeScore, ...]
    judge_loads: Tuple[JudgeLoad, ...]
    mean_aggregate: float
    balance: int
    interval: MeanInterval

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "assignments": [list(a) for a in self.assignments],
            "per_item": [s.to_dict() for s in self.per_item],
            "judge_loads": [load.to_dict() for load in self.judge_loads],
            "mean_aggregate": self.mean_aggregate,
            "balance": self.balance,
            "interval": self.interval.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────
# Allocation
# ─────────────────────────────────────────────────────────────────────────


def cyclic_allocate(n_items: int, n_judges: int, offset: int = 0) -> List[int]:
    """Round-robin judge indices for ``n_items`` items.

    Item ``i`` is routed to judge ``(i + offset) % n_judges``. This balances
    every judge across the batch to within one item, so systematic per-judge
    offsets cancel in the batch mean rather than accumulate.

    Parameters
    ----------
    n_items:
        Number of items to route; must be ``>= 0``.
    n_judges:
        Size of the judge panel; must be ``>= 1``.
    offset:
        Rotation offset — lets successive batches start on a different judge
        so no single judge is permanently assigned the "first" item.

    Returns
    -------
    list[int]
        Judge index for each item, in item order.
    """
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    if n_judges < 1:
        raise ValueError("n_judges must be >= 1")
    return [(i + offset) % n_judges for i in range(n_items)]


# ─────────────────────────────────────────────────────────────────────────
# Coverage-correct interval
# ─────────────────────────────────────────────────────────────────────────


def batch_mean_interval(
    aggregates: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
) -> MeanInterval:
    """Student-t confidence interval over independent per-item aggregates.

    The independent sampling unit for "true mean quality" is the **item**,
    not the criterion. Building the interval here — rather than over the
    criteria of a single item — is what gives it valid empirical coverage
    (Causal Judge Evaluation, arXiv:2512.11150).

    Parameters
    ----------
    aggregates:
        Per-item aggregate scores; must be non-empty. With a single item the
        interval is degenerate (``lo == hi == mean``) since variance is
        undefined.
    confidence:
        Nominal two-sided coverage in ``(0, 1)``; default ``0.95``.

    Returns
    -------
    MeanInterval
    """
    if not aggregates:
        raise ValueError("aggregates must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    n = len(aggregates)
    mean = statistics.fmean(aggregates)
    if n < 2:
        return MeanInterval(
            mean=mean, lo=mean, hi=mean, n=n, stdev=0.0, confidence=confidence
        )

    stdev = statistics.stdev(aggregates)  # ddof = 1
    se = stdev / (n**0.5)
    t_crit = _student_t_critical(n - 1, 1.0 - confidence)
    half = t_crit * se
    return MeanInterval(
        mean=mean,
        lo=mean - half,
        hi=mean + half,
        n=n,
        stdev=stdev,
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────
# Variance decomposition
# ─────────────────────────────────────────────────────────────────────────


def _validate_grid(grid: Mapping[str, Sequence[float]]) -> int:
    """Validate a judge × item grid and return the shared item count."""
    if len(grid) < 2:
        raise ValueError("variance decomposition needs at least 2 judges")
    lengths = {len(row) for row in grid.values()}
    if len(lengths) != 1:
        raise ValueError("every judge must score the same number of items")
    n_items = lengths.pop()
    if n_items < 2:
        raise ValueError("variance decomposition needs at least 2 items")
    return n_items


def _ss_from_means(means: Sequence[float], grand: float, weight: int) -> float:
    """Weighted sum of squared deviations of group means from the grand mean."""
    return weight * sum((m - grand) ** 2 for m in means)


def _fraction(numerator: float, total: float) -> float:
    """``numerator / total``; ``0.0`` when ``total`` is zero (flat grid)."""
    return numerator / total if total > 0 else 0.0


def variance_components(grid: Mapping[str, Sequence[float]]) -> VarianceComponents:
    """Decompose a judge × item score grid into judge / item / residual SS.

    Expects one observation per cell (every judge scores every item once).
    Uses the standard two-way additive ANOVA identity::

        SS_total = SS_judge + SS_item + SS_residual

    Parameters
    ----------
    grid:
        ``{judge_name: [score_for_item_0, score_for_item_1, ...]}``. Needs at
        least two judges and two items, with equal-length rows.

    Returns
    -------
    VarianceComponents
    """
    n_items = _validate_grid(grid)
    judges = list(grid)
    n_judges = len(judges)

    all_scores = [s for row in grid.values() for s in row]
    grand = statistics.fmean(all_scores)

    judge_means = [statistics.fmean(grid[j]) for j in judges]
    item_means = [
        statistics.fmean([grid[j][i] for j in judges]) for i in range(n_items)
    ]

    ss_judge = _ss_from_means(judge_means, grand, n_items)
    ss_item = _ss_from_means(item_means, grand, n_judges)
    ss_total = sum((s - grand) ** 2 for s in all_scores)
    ss_residual = max(0.0, ss_total - ss_judge - ss_item)

    return VarianceComponents(
        ss_judge=ss_judge,
        ss_item=ss_item,
        ss_residual=ss_residual,
        ss_total=ss_total,
        judge_variance_fraction=_fraction(ss_judge, ss_total),
        item_variance_fraction=_fraction(ss_item, ss_total),
        n_judges=n_judges,
        n_items=n_items,
    )


# ─────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────


def _judge_loads(
    per_item: Sequence[JudgeScore],
    judges: Sequence[JudgeConfig],
) -> Tuple[List[JudgeLoad], int]:
    """Compute per-judge load + the balance spread across the panel."""
    by_judge: Dict[str, List[float]] = {j.name: [] for j in judges}
    for score in per_item:
        by_judge.setdefault(score.judge, []).append(score.aggregate)

    loads = [
        JudgeLoad(
            judge=name,
            n_items=len(aggs),
            mean_aggregate=statistics.fmean(aggs) if aggs else 0.0,
        )
        for name, aggs in by_judge.items()
    ]
    counts = [load.n_items for load in loads]
    balance = max(counts) - min(counts) if counts else 0
    return loads, balance


def cyclic_evaluate(
    items: Sequence[Tuple[str, str]],
    judges: Sequence[JudgeConfig],
    *,
    offset: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> CyclicEvalReport:
    """Score a batch of items with round-robin judge allocation.

    Each item is scored by exactly one judge — rotated round-robin — for
    single-judge cost while balancing every judge across the batch. The
    report carries the coverage-correct interval on the batch mean and the
    per-judge load balance.

    Parameters
    ----------
    items:
        Non-empty sequence of ``(prompt, response)`` pairs.
    judges:
        Non-empty judge panel.
    offset:
        Round-robin rotation offset (see :func:`cyclic_allocate`).
    confidence:
        Coverage level for the batch-mean interval.

    Returns
    -------
    CyclicEvalReport
    """
    if not items:
        raise ValueError("items must be non-empty")
    if not judges:
        raise ValueError("judges must be non-empty")

    allocation = cyclic_allocate(len(items), len(judges), offset)
    per_item: List[JudgeScore] = []
    assignments: List[Tuple[int, str]] = []
    for i, (prompt, response) in enumerate(items):
        judge = judges[allocation[i]]
        per_item.append(judge_evaluate(prompt, response, judge))
        assignments.append((i, judge.name))

    aggregates = [s.aggregate for s in per_item]
    loads, balance = _judge_loads(per_item, judges)
    interval = batch_mean_interval(aggregates, confidence)

    return CyclicEvalReport(
        assignments=tuple(assignments),
        per_item=tuple(per_item),
        judge_loads=tuple(loads),
        mean_aggregate=statistics.fmean(aggregates),
        balance=balance,
        interval=interval,
    )


def full_grid_scores(
    items: Sequence[Tuple[str, str]],
    judges: Sequence[JudgeConfig],
) -> Dict[str, List[float]]:
    """Score every item with every judge — the grid for :func:`variance_components`.

    This is the expensive N·K reference run. Use it on a small calibration
    slice to measure ``judge_variance_fraction`` and decide whether the cheap
    round-robin :func:`cyclic_evaluate` is safe for the full sweep.

    Parameters
    ----------
    items:
        Non-empty sequence of ``(prompt, response)`` pairs.
    judges:
        Non-empty judge panel.

    Returns
    -------
    dict[str, list[float]]
        ``{judge_name: [aggregate_for_item_0, ...]}``.
    """
    if not items:
        raise ValueError("items must be non-empty")
    if not judges:
        raise ValueError("judges must be non-empty")

    grid: Dict[str, List[float]] = {}
    for judge in judges:
        grid[judge.name] = [
            judge_evaluate(prompt, response, judge).aggregate
            for prompt, response in items
        ]
    return grid


__all__ = [
    "DEFAULT_CONFIDENCE",
    "MeanInterval",
    "JudgeLoad",
    "VarianceComponents",
    "CyclicEvalReport",
    "cyclic_allocate",
    "batch_mean_interval",
    "variance_components",
    "cyclic_evaluate",
    "full_grid_scores",
]
