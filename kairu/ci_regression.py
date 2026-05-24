"""CI regression gating for evaluation pipelines.

The use case is concrete: a release engineer wants to ship a new model,
prompt template, or system-prompt change and needs the build to fail if
any rubric criterion regresses on a curated set of golden inputs.

Workflow
--------
1.  Run ``snapshot_baseline(items, ...)`` once on a known-good
    configuration. Save the returned :class:`BaselineSnapshot` (the
    :class:`FileBaselineStore` writes it as JSON).
2.  On each CI run, score the same inputs against the new candidate and
    pass the result to :func:`check_against_baseline`. The
    :class:`RegressionReport` carries ``passed: bool`` — CI keys on that
    to decide exit code 0 vs 1.

A *regression* is any per-item criterion whose score drops by more than
``threshold`` (default ``0.05``). Five points of degradation on any
criterion of any item is the level CI typically wants to gate at — small
enough to catch real losses, large enough to ride out heuristic noise.

Items are matched by SHA-256 of the input string. Re-ordering between
runs is fine; missing or new items surface in ``unmatched_baseline`` /
``unmatched_current`` so CI can diff the corpus too.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kairu.evaluation import Evaluation, evaluate


DEFAULT_REGRESSION_THRESHOLD: float = 0.05


def _hash_input(text: str) -> str:
    """SHA-256 prefix of the input — stable across runs."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineItem:
    """One golden datapoint inside a snapshot."""

    input_hash: str
    aggregate: float
    scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "aggregate": self.aggregate,
            "scores": dict(self.scores),
        }


@dataclass(frozen=True)
class BaselineSnapshot:
    """An immutable record of a known-good evaluation pass."""

    snapshot_id: str
    created_utc: float
    rubric_name: str
    rubric_version: str
    judge_model: str
    n_items: int
    mean_aggregate: float
    items: Tuple[BaselineItem, ...]
    label: str = ""  # optional human label ("v2.3 prod", "main@abc123")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_utc": self.created_utc,
            "rubric_name": self.rubric_name,
            "rubric_version": self.rubric_version,
            "judge_model": self.judge_model,
            "n_items": self.n_items,
            "mean_aggregate": self.mean_aggregate,
            "label": self.label,
            "items": [i.to_dict() for i in self.items],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BaselineSnapshot":
        return cls(
            snapshot_id=str(d["snapshot_id"]),
            created_utc=float(d["created_utc"]),
            rubric_name=str(d["rubric_name"]),
            rubric_version=str(d.get("rubric_version", "1.0.0")),
            judge_model=str(d.get("judge_model", "kairu-heuristic-v1")),
            n_items=int(d["n_items"]),
            mean_aggregate=float(d["mean_aggregate"]),
            label=str(d.get("label", "")),
            items=tuple(
                BaselineItem(
                    input_hash=str(i["input_hash"]),
                    aggregate=float(i["aggregate"]),
                    scores={k: float(v) for k, v in i["scores"].items()},
                )
                for i in d["items"]
            ),
        )


@dataclass(frozen=True)
class CriterionRegression:
    """One detected regression — per-item, per-criterion."""

    item_idx: int
    input_hash: str
    criterion: str
    baseline_score: float
    current_score: float
    delta: float  # negative when regressed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionReport:
    """Output of :func:`check_against_baseline` — drives CI pass/fail."""

    snapshot_id: str
    n_baseline: int
    n_current: int
    n_matched: int
    threshold: float
    passed: bool
    regressions: Tuple[CriterionRegression, ...]
    mean_baseline_aggregate: float
    mean_current_aggregate: float
    mean_delta: float
    unmatched_baseline: Tuple[
        str, ...
    ]  # input_hash present in baseline but not current
    unmatched_current: Tuple[str, ...]  # input_hash present in current but not baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "n_baseline": self.n_baseline,
            "n_current": self.n_current,
            "n_matched": self.n_matched,
            "threshold": self.threshold,
            "passed": self.passed,
            "regressions": [r.to_dict() for r in self.regressions],
            "mean_baseline_aggregate": self.mean_baseline_aggregate,
            "mean_current_aggregate": self.mean_current_aggregate,
            "mean_delta": self.mean_delta,
            "unmatched_baseline": list(self.unmatched_baseline),
            "unmatched_current": list(self.unmatched_current),
        }


# ─────────────────────────────────────────────────────────────────────────
# Snapshot construction
# ─────────────────────────────────────────────────────────────────────────


def _score_items(
    items: Sequence[Mapping[str, str]],
    *,
    rubric: Optional[str],
    criteria: Optional[Sequence[str]],
    weights: Optional[Mapping[str, float]],
) -> List[Tuple[str, Evaluation]]:
    """Run ``evaluate`` over an item list; returns ``(input_hash, evaluation)``."""
    out: List[Tuple[str, Evaluation]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("each item must be a mapping with 'input' and 'output'")
        ipt = item.get("input")
        out_resp = item.get("output")
        if not isinstance(ipt, str) or not isinstance(out_resp, str):
            raise TypeError("each item must have string 'input' and 'output' fields")
        ev = evaluate(
            ipt,
            out_resp,
            rubric=rubric,
            criteria=list(criteria) if criteria else None,
            weights=weights,
        )
        out.append((_hash_input(ipt), ev))
    return out


def snapshot_baseline(
    items: Sequence[Mapping[str, str]],
    *,
    rubric: Optional[str] = None,
    criteria: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
    judge_model: str = "kairu-heuristic-v1",
    label: str = "",
) -> BaselineSnapshot:
    """Score the items and freeze the result as a baseline."""
    if not items:
        raise ValueError("items must be non-empty")
    scored = _score_items(items, rubric=rubric, criteria=criteria, weights=weights)
    baseline_items: List[BaselineItem] = []
    for h, ev in scored:
        baseline_items.append(
            BaselineItem(
                input_hash=h,
                aggregate=ev.aggregate,
                scores={cs.name: cs.score for cs in ev.scores},
            )
        )
    mean_agg = sum(i.aggregate for i in baseline_items) / len(baseline_items)
    # Resolve the rubric name + version from the first evaluation (all share one).
    first_ev = scored[0][1]
    rubric_name = first_ev.rubric
    # Look up version from the registry; fall back gracefully if not registered.
    try:
        from kairu.evaluation import RUBRICS

        rubric_version = getattr(RUBRICS.get(rubric_name), "version", "1.0.0")
    except Exception:  # noqa: BLE001 — defensive
        rubric_version = "1.0.0"
    return BaselineSnapshot(
        snapshot_id=uuid.uuid4().hex,
        created_utc=time.time(),
        rubric_name=rubric_name,
        rubric_version=rubric_version,
        judge_model=judge_model,
        n_items=len(baseline_items),
        mean_aggregate=mean_agg,
        items=tuple(baseline_items),
        label=label,
    )


# ─────────────────────────────────────────────────────────────────────────
# Regression check
# ─────────────────────────────────────────────────────────────────────────


def check_against_baseline(
    baseline: BaselineSnapshot,
    items: Sequence[Mapping[str, str]],
    *,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
    rubric: Optional[str] = None,
    criteria: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> RegressionReport:
    """Score ``items`` and compare each criterion against the baseline.

    A regression is recorded when ``baseline_score - current_score > threshold``.
    Items are matched by SHA-256 of the input string — order is not required.
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    eff_rubric = rubric if rubric is not None else baseline.rubric_name
    scored = _score_items(items, rubric=eff_rubric, criteria=criteria, weights=weights)
    baseline_by_hash = {b.input_hash: b for b in baseline.items}
    current_by_hash: Dict[str, Evaluation] = {h: ev for h, ev in scored}

    matched_hashes = sorted(baseline_by_hash.keys() & current_by_hash.keys())
    unmatched_baseline = tuple(sorted(baseline_by_hash.keys() - current_by_hash.keys()))
    unmatched_current = tuple(sorted(current_by_hash.keys() - baseline_by_hash.keys()))

    regressions: List[CriterionRegression] = []
    base_aggs: List[float] = []
    cur_aggs: List[float] = []
    # Use insertion order from current items for stable indices.
    hash_to_current_idx = {h: idx for idx, (h, _) in enumerate(scored)}
    for h in matched_hashes:
        b = baseline_by_hash[h]
        c = current_by_hash[h]
        base_aggs.append(b.aggregate)
        cur_aggs.append(c.aggregate)
        cur_scores = {cs.name: cs.score for cs in c.scores}
        for crit, b_score in b.scores.items():
            if crit not in cur_scores:
                continue
            c_score = cur_scores[crit]
            delta = c_score - b_score
            if -delta > threshold:  # i.e. baseline - current > threshold
                regressions.append(
                    CriterionRegression(
                        item_idx=hash_to_current_idx[h],
                        input_hash=h,
                        criterion=crit,
                        baseline_score=b_score,
                        current_score=c_score,
                        delta=delta,
                    )
                )

    n_matched = len(matched_hashes)
    mean_base = sum(base_aggs) / n_matched if n_matched else 0.0
    mean_cur = sum(cur_aggs) / n_matched if n_matched else 0.0
    mean_delta = mean_cur - mean_base

    return RegressionReport(
        snapshot_id=baseline.snapshot_id,
        n_baseline=baseline.n_items,
        n_current=len(scored),
        n_matched=n_matched,
        threshold=threshold,
        passed=len(regressions) == 0 and len(unmatched_baseline) == 0,
        regressions=tuple(regressions),
        mean_baseline_aggregate=mean_base,
        mean_current_aggregate=mean_cur,
        mean_delta=mean_delta,
        unmatched_baseline=unmatched_baseline,
        unmatched_current=unmatched_current,
    )


# ─────────────────────────────────────────────────────────────────────────
# Persistence — in-memory + filesystem stores
# ─────────────────────────────────────────────────────────────────────────


class BaselineStore:
    """In-memory snapshot store; thread-safety is the caller's job."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, BaselineSnapshot] = {}

    def save(self, snapshot: BaselineSnapshot) -> str:
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot.snapshot_id

    def load(self, snapshot_id: str) -> BaselineSnapshot:
        if snapshot_id not in self._snapshots:
            raise KeyError(snapshot_id)
        return self._snapshots[snapshot_id]

    def list(self) -> List[str]:
        # Newest first.
        return sorted(
            self._snapshots.keys(),
            key=lambda s: self._snapshots[s].created_utc,
            reverse=True,
        )

    def __len__(self) -> int:
        return len(self._snapshots)


class FileBaselineStore(BaselineStore):
    """Snapshots persist as one JSON file per snapshot under ``root``.

    Filename: ``<snapshot_id>.json``. Atomic write via tempfile + rename.
    """

    def __init__(self, root: str) -> None:
        super().__init__()
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        for p in self._root.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                snap = BaselineSnapshot.from_dict(data)
                self._snapshots[snap.snapshot_id] = snap
            except Exception:  # noqa: BLE001 — skip corrupt files
                continue

    def save(self, snapshot: BaselineSnapshot) -> str:
        sid = super().save(snapshot)
        path = self._root / f"{sid}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return sid


def open_default_store() -> BaselineStore:
    """Resolve a store from ``KAIRU_CI_DIR``; in-memory if unset."""
    root = os.environ.get("KAIRU_CI_DIR")
    if not root:
        return BaselineStore()
    return FileBaselineStore(root)


__all__ = [
    "DEFAULT_REGRESSION_THRESHOLD",
    "BaselineItem",
    "BaselineSnapshot",
    "CriterionRegression",
    "RegressionReport",
    "BaselineStore",
    "FileBaselineStore",
    "snapshot_baseline",
    "check_against_baseline",
    "open_default_store",
]
