"""Judge-ensemble evaluation — run N judge configurations, aggregate by
median, report inter-judge disagreement.

Why an ensemble of judges
-------------------------
A single judge (rubric + scorers + weights) is a single perspective on
*good*. Two reasonable judges can disagree by 30 points on a borderline
response — that disagreement itself is signal, and any production
evaluation pipeline that ignores it eventually mis-deploys.

We aggregate per-criterion with the **median** rather than the mean
because the median tolerates one outlier judge without poisoning the
result (e.g. four judges agree at 0.8 and one screams 0.2 → median 0.8,
mean 0.68). For N=3 judges the median is also the order statistic with
the lowest variance under heavy-tailed disagreement.

Disagreement is reported as per-criterion stdev. When **any** criterion's
stdev crosses ``disagreement_threshold`` (default 0.2), the result carries
``disagreement_flag = True`` — the caller's signal to defer to a human or
queue the item for a follow-up judge.

Why not Krippendorff's alpha?
-----------------------------
At N=2 judges and one item, Krippendorff's alpha is degenerate; at N=3
it is dominated by the per-criterion stdev for the simple disagreement
question we care about ("do my judges fight on this item?"). Stdev is the
honest metric for our scale.

Judge configurations as heuristic perspectives
----------------------------------------------
The library's scorers are deterministic — we cannot literally call
``gpt-4o`` and ``claude-3-5-sonnet`` and ``gemini-1.5-pro`` here. A
``JudgeConfig`` carries (rubric, criteria override, weight override,
seeded noise). When ``noise > 0`` we add deterministic seeded Gaussian
perturbation to each criterion score — this simulates inter-judge
variance authentically without sacrificing test reproducibility. When
``noise = 0`` the judge is fully deterministic, which is the right
default for CI and golden-snapshot use.

Real LLM judges plug in later behind the same ``JudgeConfig`` contract
by adding a ``model_name`` field + a dispatch table in this module — no
caller changes required.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kairu.evaluation import (
    CRITERIA,
    Evaluation,
    TIE_EPSILON,
    evaluate,
)


# Default disagreement threshold — picked to match the heuristic noise
# floor of the underlying scorers (TIE_EPSILON × 40 ≈ 0.2). One judge
# moving 0.2 from the median is the smallest gap worth surfacing.
DEFAULT_DISAGREEMENT_THRESHOLD: float = 0.2


@dataclass(frozen=True)
class JudgeConfig:
    """One judge's evaluation perspective.

    Parameters
    ----------
    name : str
        Stable identifier — surfaces in the result so the caller can
        attribute disagreement to a specific perspective.
    rubric : str | None
        Built-in or registered rubric name. ``None`` falls back to the
        default rubric (same as :func:`kairu.evaluation.evaluate`).
    criteria : tuple[str, ...] | None
        Optional override of the rubric's criteria list. Each entry must
        be a key of :data:`kairu.evaluation.CRITERIA`.
    weights : Mapping[str, float] | None
        Optional per-criterion weight override. Missing keys fall back to
        the rubric's defaults.
    seed : int | None
        Deterministic seed for the noise stream. Defaults to a hash of
        ``name`` — different judges get different streams without callers
        having to think about it.
    noise : float
        Standard deviation of additive Gaussian noise on each criterion
        score (clamped to [0, 1] after). 0.0 → fully deterministic. Real
        LLM judges typically sit around 0.05–0.08 across reruns at temp 0.
    """

    name: str
    rubric: Optional[str] = "default"
    criteria: Optional[Tuple[str, ...]] = None
    weights: Optional[Mapping[str, float]] = None
    seed: Optional[int] = None
    noise: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("JudgeConfig.name must be a non-empty string")
        if self.noise < 0.0:
            raise ValueError("JudgeConfig.noise must be >= 0")
        if self.criteria is not None:
            unknown = [c for c in self.criteria if c not in CRITERIA]
            if unknown:
                raise ValueError(f"unknown criteria: {unknown}")


@dataclass(frozen=True)
class JudgeScore:
    """One judge's verdict on one (prompt, response) pair."""

    judge: str
    rubric: str
    aggregate: float
    scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge": self.judge,
            "rubric": self.rubric,
            "aggregate": self.aggregate,
            "scores": dict(self.scores),
        }


@dataclass(frozen=True)
class EnsembleResult:
    """Aggregate of N judges on one (prompt, response) pair."""

    judges: Tuple[JudgeScore, ...]
    median_scores: Dict[str, float]
    stdev_scores: Dict[str, float]
    median_aggregate: float
    max_disagreement: float
    disagreement_flag: bool
    disagreement_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judges": [j.to_dict() for j in self.judges],
            "median_scores": dict(self.median_scores),
            "stdev_scores": dict(self.stdev_scores),
            "median_aggregate": self.median_aggregate,
            "max_disagreement": self.max_disagreement,
            "disagreement_flag": self.disagreement_flag,
            "disagreement_threshold": self.disagreement_threshold,
        }


@dataclass(frozen=True)
class EnsembleComparison:
    """A/B comparison aggregated across N judges."""

    a: EnsembleResult
    b: EnsembleResult
    median_diff: float
    winner: str  # "a" | "b" | "tie"
    per_criterion: Dict[str, Dict[str, float]]
    disagreement_flag: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "median_diff": self.median_diff,
            "winner": self.winner,
            "per_criterion": {k: dict(v) for k, v in self.per_criterion.items()},
            "disagreement_flag": self.disagreement_flag,
        }


# ─────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────


def _seed_for(judge: JudgeConfig) -> int:
    """Deterministic seed: explicit if given, else a hash of the name."""
    if judge.seed is not None:
        return int(judge.seed) & 0xFFFF_FFFF
    digest = hashlib.sha256(judge.name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _perturb(score: float, rng: random.Random, noise: float) -> float:
    """Add seeded Gaussian noise, clamped to [0, 1]."""
    if noise <= 0.0:
        return score
    perturbed = score + rng.gauss(0.0, noise)
    return max(0.0, min(1.0, perturbed))


def judge_evaluate(prompt: str, response: str, judge: JudgeConfig) -> JudgeScore:
    """Score one (prompt, response) pair through one judge.

    Uses :func:`kairu.evaluation.evaluate` under the hood, then applies the
    judge's deterministic noise stream. The aggregate is recomputed from
    the post-noise scores so it remains internally consistent.
    """
    ev: Evaluation = evaluate(
        prompt,
        response,
        rubric=judge.rubric,
        criteria=list(judge.criteria) if judge.criteria else None,
        weights=judge.weights,
    )
    if judge.noise == 0.0:
        scores = {cs.name: cs.score for cs in ev.scores}
        return JudgeScore(
            judge=judge.name,
            rubric=ev.rubric,
            aggregate=ev.aggregate,
            scores=scores,
        )

    rng = random.Random(_seed_for(judge))
    perturbed: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for cs in ev.scores:
        perturbed[cs.name] = _perturb(cs.score, rng, judge.noise)
        weights[cs.name] = cs.weight
    total_w = sum(weights.values())
    if total_w <= 0.0:
        agg = 0.0
    else:
        agg = sum(perturbed[k] * weights[k] for k in perturbed) / total_w
    return JudgeScore(
        judge=judge.name,
        rubric=ev.rubric,
        aggregate=agg,
        scores=perturbed,
    )


def _aggregate_judges(
    per_judge: Sequence[JudgeScore],
    *,
    disagreement_threshold: float,
) -> EnsembleResult:
    """Stack per-judge scores into the ensemble verdict."""
    if not per_judge:
        raise ValueError("at least one judge score is required")
    criterion_names: List[str] = list(per_judge[0].scores.keys())
    median_scores: Dict[str, float] = {}
    stdev_scores: Dict[str, float] = {}
    for name in criterion_names:
        column = [j.scores[name] for j in per_judge if name in j.scores]
        if not column:
            continue
        median_scores[name] = statistics.median(column)
        stdev_scores[name] = statistics.pstdev(column) if len(column) > 1 else 0.0
    aggregates = [j.aggregate for j in per_judge]
    median_aggregate = statistics.median(aggregates)
    max_dis = max(stdev_scores.values()) if stdev_scores else 0.0
    return EnsembleResult(
        judges=tuple(per_judge),
        median_scores=median_scores,
        stdev_scores=stdev_scores,
        median_aggregate=median_aggregate,
        max_disagreement=max_dis,
        disagreement_flag=max_dis > disagreement_threshold,
        disagreement_threshold=disagreement_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────


def ensemble_evaluate(
    prompt: str,
    response: str,
    judges: Sequence[JudgeConfig],
    *,
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> EnsembleResult:
    """Run every judge against one (prompt, response) pair."""
    if not judges:
        raise ValueError("judges must be a non-empty sequence")
    per_judge = [judge_evaluate(prompt, response, j) for j in judges]
    return _aggregate_judges(per_judge, disagreement_threshold=disagreement_threshold)


def ensemble_compare(
    prompt: str,
    response_a: str,
    response_b: str,
    judges: Sequence[JudgeConfig],
    *,
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
    tie_epsilon: float = TIE_EPSILON,
) -> EnsembleComparison:
    """A/B comparison aggregated across N judges."""
    a = ensemble_evaluate(
        prompt, response_a, judges, disagreement_threshold=disagreement_threshold
    )
    b = ensemble_evaluate(
        prompt, response_b, judges, disagreement_threshold=disagreement_threshold
    )
    median_diff = a.median_aggregate - b.median_aggregate
    if abs(median_diff) <= tie_epsilon:
        winner = "tie"
    elif median_diff > 0:
        winner = "a"
    else:
        winner = "b"
    per_criterion: Dict[str, Dict[str, float]] = {}
    for name in a.median_scores:
        if name not in b.median_scores:
            continue
        am, bm = a.median_scores[name], b.median_scores[name]
        diff = am - bm
        if abs(diff) <= tie_epsilon:
            w = "tie"
        elif diff > 0:
            w = "a"
        else:
            w = "b"
        per_criterion[name] = {
            "a_median": am,
            "b_median": bm,
            "diff": diff,
            "a_stdev": a.stdev_scores.get(name, 0.0),
            "b_stdev": b.stdev_scores.get(name, 0.0),
            "winner": w,
        }
    return EnsembleComparison(
        a=a,
        b=b,
        median_diff=median_diff,
        winner=winner,
        per_criterion=per_criterion,
        disagreement_flag=a.disagreement_flag or b.disagreement_flag,
    )


__all__ = [
    "DEFAULT_DISAGREEMENT_THRESHOLD",
    "JudgeConfig",
    "JudgeScore",
    "EnsembleResult",
    "EnsembleComparison",
    "judge_evaluate",
    "ensemble_evaluate",
    "ensemble_compare",
]
