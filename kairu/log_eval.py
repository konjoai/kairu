"""Production-log → evaluation pipeline.

Take a JSONL stream of ``{input, output}`` records from your application's
inference log and run them through a kairu rubric. Returns aggregate
statistics plus a per-item breakdown ready for a dashboard or a CI gate.

Designed to drop into a Kubernetes CronJob or a CI step:

    items = [json.loads(l) for l in open("inference.log")]
    report = evaluate_log(items, rubric="helpfulness", threshold=0.65)
    sys.exit(0 if report.passed else 1)

The threshold is a pass gate on ``mean_aggregate`` — the most common
ask is "block the deploy if quality dropped overall." Per-item failure
counts come along for free in ``n_failed`` (items whose own aggregate
sits below the same threshold) so the caller can also gate on per-item
SLOs.

Items can carry arbitrary ``metadata`` (request id, region, model tag)
which we pass through untouched in ``per_item`` — useful for slicing
the report by deployment cohort downstream.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kairu.evaluation import Evaluation, evaluate


DEFAULT_LOG_THRESHOLD: float = 0.5


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogItemResult:
    """Per-item slice of the report."""

    idx: int
    input_hash: str
    aggregate: float
    scores: Dict[str, float]
    passed: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "input_hash": self.input_hash,
            "aggregate": self.aggregate,
            "scores": dict(self.scores),
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LogEvalReport:
    """Top-level output of :func:`evaluate_log`."""

    n_items: int
    rubric_name: str
    rubric_version: str
    judge_model: str
    threshold: float
    passed: bool  # gate on mean_aggregate
    n_failed: int  # count of items with aggregate < threshold
    mean_aggregate: float
    median_aggregate: float
    min_aggregate: float
    max_aggregate: float
    stdev_aggregate: float
    per_criterion_mean: Dict[str, float]
    per_criterion_min: Dict[str, float]
    items: Tuple[LogItemResult, ...]
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_items": self.n_items,
            "rubric_name": self.rubric_name,
            "rubric_version": self.rubric_version,
            "judge_model": self.judge_model,
            "threshold": self.threshold,
            "passed": self.passed,
            "n_failed": self.n_failed,
            "mean_aggregate": self.mean_aggregate,
            "median_aggregate": self.median_aggregate,
            "min_aggregate": self.min_aggregate,
            "max_aggregate": self.max_aggregate,
            "stdev_aggregate": self.stdev_aggregate,
            "per_criterion_mean": dict(self.per_criterion_mean),
            "per_criterion_min": dict(self.per_criterion_min),
            "items": [i.to_dict() for i in self.items],
            "duration_seconds": self.duration_seconds,
        }


# ─────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────


def evaluate_log(
    items: Sequence[Mapping[str, Any]],
    *,
    rubric: Optional[str] = None,
    criteria: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
    threshold: float = DEFAULT_LOG_THRESHOLD,
    judge_model: str = "kairu-heuristic-v1",
) -> LogEvalReport:
    """Batch-evaluate inference log records.

    Each item must have string ``input`` and ``output`` fields; any other
    keys are treated as opaque metadata and pass through to ``per_item``.
    """
    if not items:
        raise ValueError("items must be non-empty")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")

    t0 = time.monotonic()
    results: List[LogItemResult] = []
    aggregates: List[float] = []
    crit_columns: Dict[str, List[float]] = {}
    rubric_name = ""
    n_failed = 0

    for idx, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise TypeError(f"item {idx}: must be a mapping")
        ipt = raw.get("input")
        out = raw.get("output")
        if not isinstance(ipt, str) or not isinstance(out, str):
            raise TypeError(f"item {idx}: 'input' and 'output' must be strings")

        ev: Evaluation = evaluate(
            ipt,
            out,
            rubric=rubric,
            criteria=list(criteria) if criteria else None,
            weights=weights,
        )
        if not rubric_name:
            rubric_name = ev.rubric

        scores = {cs.name: cs.score for cs in ev.scores}
        for name, score in scores.items():
            crit_columns.setdefault(name, []).append(score)
        aggregates.append(ev.aggregate)
        item_passed = ev.aggregate >= threshold
        if not item_passed:
            n_failed += 1

        # Pull metadata off opaque keys (anything that isn't input/output).
        metadata = {k: v for k, v in raw.items() if k not in ("input", "output")}
        results.append(
            LogItemResult(
                idx=idx,
                input_hash=_hash(ipt),
                aggregate=ev.aggregate,
                scores=scores,
                passed=item_passed,
                metadata=metadata,
            )
        )

    mean_agg = statistics.fmean(aggregates)
    median_agg = statistics.median(aggregates)
    min_agg = min(aggregates)
    max_agg = max(aggregates)
    stdev_agg = statistics.pstdev(aggregates) if len(aggregates) > 1 else 0.0
    per_crit_mean = {k: statistics.fmean(v) for k, v in crit_columns.items()}
    per_crit_min = {k: min(v) for k, v in crit_columns.items()}

    # Resolve rubric version from the registry (defensive fallback).
    try:
        from kairu.evaluation import RUBRICS

        rubric_version = getattr(RUBRICS.get(rubric_name), "version", "1.0.0")
    except Exception:  # noqa: BLE001
        rubric_version = "1.0.0"

    return LogEvalReport(
        n_items=len(results),
        rubric_name=rubric_name,
        rubric_version=rubric_version,
        judge_model=judge_model,
        threshold=threshold,
        passed=mean_agg >= threshold,
        n_failed=n_failed,
        mean_aggregate=mean_agg,
        median_aggregate=median_agg,
        min_aggregate=min_agg,
        max_aggregate=max_agg,
        stdev_aggregate=stdev_agg,
        per_criterion_mean=per_crit_mean,
        per_criterion_min=per_crit_min,
        items=tuple(results),
        duration_seconds=time.monotonic() - t0,
    )


__all__ = [
    "DEFAULT_LOG_THRESHOLD",
    "LogItemResult",
    "LogEvalReport",
    "evaluate_log",
]
