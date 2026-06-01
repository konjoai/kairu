"""Real leaderboard endpoint — proper SQLite-backed score history.

The demo UI's "Leaderboard" tab was synthesising ranks from raw audit-log
rows since v0.19. That works as a stop-gap but doesn't compose well: every
client redoes the aggregation; the audit table grows without bound; and
the audit log captures *judge identity*, not the *model being evaluated*.

This module owns a dedicated table keyed on the model identity supplied
by the caller (or derived from request metadata when absent). Every
``/evaluate`` or ``/evaluate/ensemble`` call that carries a ``model``
field auto-appends one row. Rank queries hit a single indexed table.

Schema
------
``leaderboard_entries(id, model, prompt_hash, rubric_name, aggregate,
criteria_json, timestamp_utc)`` — one row per scored ``(model, prompt)``.
Indexed on ``(model, timestamp_utc)`` for the common period-bounded query.

Rank query
----------
``rank(metric, days, limit, criterion=None)`` returns rows shaped::

    {
      "rank":            int,
      "model":           str,
      "mean_score":      float,    # mean of metric over the period
      "n_evaluations":   int,
      "delta":           float,    # mean_score - mean over the prior period
      "trend":           [float],  # last 10 scores chronologically
      "p25"/"p50"/"p75": float,    # quartiles over the period
    }

``metric`` is either ``"aggregate"`` or a criterion name. When the
criterion is missing from a row's ``criteria_json``, the row is excluded
from that metric's ranking — same semantics as the audit log.

Delta interpretation
--------------------
Comparing to *the immediately prior period of the same length* answers
the question ranking views actually care about: "is this model improving
or regressing right now?" If the period is "last 7 days", the prior
period is the 7 days before that. When fewer than 2 evaluations exist
across both periods, delta is 0.0 (insufficient signal).
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kairu.audit import hash_inputs


_SCHEMA = """
CREATE TABLE IF NOT EXISTS leaderboard_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model          TEXT    NOT NULL,
    prompt_hash    TEXT    NOT NULL,
    rubric_name    TEXT    NOT NULL,
    aggregate      REAL    NOT NULL,
    criteria_json  TEXT    NOT NULL DEFAULT '{}',
    timestamp_utc  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lb_model_ts ON leaderboard_entries(model, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_lb_ts       ON leaderboard_entries(timestamp_utc);
"""


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _iso_to_dt(iso: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp produced by :func:`_now_iso_utc`."""
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso)


# ─────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LeaderboardEntry:
    """One row in the leaderboard table."""

    id: int
    model: str
    prompt_hash: str
    rubric_name: str
    aggregate: float
    criteria: Dict[str, float]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "rubric_name": self.rubric_name,
            "aggregate": self.aggregate,
            "criteria": dict(self.criteria),
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass(frozen=True)
class LeaderboardRow:
    """One entry in the ranked output of :meth:`LeaderboardStore.rank`."""

    rank: int
    model: str
    mean_score: float
    n_evaluations: int
    delta: float
    trend: List[float]
    p25: float
    p50: float
    p75: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "model": self.model,
            "mean_score": self.mean_score,
            "n_evaluations": self.n_evaluations,
            "delta": self.delta,
            "trend": list(self.trend),
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
        }


# ─────────────────────────────────────────────────────────────────────────
# Storage + ranking
# ─────────────────────────────────────────────────────────────────────────


class LeaderboardStore:
    """Thread-safe SQLite-backed leaderboard store.

    Pass ``:memory:`` for ephemeral storage (tests) or a filesystem path
    for durable storage. Multiple workers pointing at the same file are
    fine for the read-heavy workload — SQLite WAL handles concurrent
    rankers cleanly.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> str:
        return self._path

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM leaderboard_entries"
            ).fetchone()["n"])

    def record(
        self,
        *,
        model: str,
        prompt: str,
        rubric_name: str,
        aggregate: float,
        criteria: Optional[Mapping[str, float]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> int:
        """Append one row. Returns the new row id."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not (0.0 <= aggregate <= 1.0):
            raise ValueError("aggregate must be in [0, 1]")
        ts = timestamp_utc or _now_iso_utc()
        ph = hash_inputs(prompt, "")
        cj = json.dumps(dict(criteria or {}), sort_keys=True, ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO leaderboard_entries "
                "(model, prompt_hash, rubric_name, aggregate, criteria_json, timestamp_utc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (model, ph, rubric_name, float(aggregate), cj, ts),
            )
            return int(cur.lastrowid)

    def list_models(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT model FROM leaderboard_entries ORDER BY model"
            ).fetchall()
        return [r["model"] for r in rows]

    def rank(
        self,
        *,
        metric: str = "aggregate",
        days: Optional[int] = None,
        limit: int = 20,
        criterion: Optional[str] = None,
    ) -> List[LeaderboardRow]:
        """Rank models by mean score over the requested period.

        ``metric == "aggregate"`` (default) keys on the row's ``aggregate``
        column. Any other value is treated as a criterion name and is
        looked up inside ``criteria_json``; rows missing that key are
        silently excluded from that metric's ranking.

        ``days = None`` means all-time; otherwise only rows with
        ``timestamp_utc`` newer than ``now - days`` are considered. The
        delta is computed against the *prior* period of the same length.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        now = datetime.now(timezone.utc)
        cur_cutoff: Optional[datetime] = None
        prev_lower: Optional[datetime] = None
        prev_upper: Optional[datetime] = None
        if days is not None:
            cur_cutoff = now - timedelta(days=days)
            prev_upper = cur_cutoff
            prev_lower = cur_cutoff - timedelta(days=days)

        with self._lock:
            rows = self._conn.execute(
                "SELECT model, aggregate, criteria_json, timestamp_utc "
                "FROM leaderboard_entries ORDER BY timestamp_utc ASC"
            ).fetchall()

        # Group by model and split into current / prior buckets.
        scores_current: Dict[str, List[tuple[datetime, float]]] = {}
        scores_prior:   Dict[str, List[tuple[datetime, float]]] = {}
        for r in rows:
            model = r["model"]
            try:
                ts = _iso_to_dt(r["timestamp_utc"])
            except Exception:
                continue
            if metric == "aggregate":
                val = float(r["aggregate"])
            else:
                try:
                    val = float(json.loads(r["criteria_json"]).get(metric, None))
                except (ValueError, TypeError):
                    continue
            # Period filter.
            in_current = cur_cutoff is None or ts >= cur_cutoff
            in_prior   = (prev_lower is not None
                          and prev_upper is not None
                          and prev_lower <= ts < prev_upper)
            if in_current:
                scores_current.setdefault(model, []).append((ts, val))
            elif in_prior:
                scores_prior.setdefault(model, []).append((ts, val))

        rankings: List[LeaderboardRow] = []
        for model, samples in scores_current.items():
            if not samples:
                continue
            samples.sort(key=lambda x: x[0])
            values = [v for _, v in samples]
            mean = statistics.fmean(values)
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            p = lambda q: sorted_vals[min(n - 1, int(q * n))] if n else 0.0
            # Delta vs prior period (0 when one of them is empty).
            prior_vals = [v for _, v in scores_prior.get(model, [])]
            delta = mean - statistics.fmean(prior_vals) if prior_vals else 0.0
            rankings.append(LeaderboardRow(
                rank=0,  # filled after sort
                model=model,
                mean_score=mean,
                n_evaluations=len(values),
                delta=delta,
                trend=values[-10:],
                p25=p(0.25),
                p50=p(0.50),
                p75=p(0.75),
            ))

        rankings.sort(key=lambda r: (-r.mean_score, -r.n_evaluations, r.model))
        rankings = rankings[:limit]
        # Stamp ranks now that the order is final.
        rankings = [LeaderboardRow(
            rank=i + 1,
            model=r.model, mean_score=r.mean_score, n_evaluations=r.n_evaluations,
            delta=r.delta, trend=r.trend, p25=r.p25, p50=r.p50, p75=r.p75,
        ) for i, r in enumerate(rankings)]
        return rankings

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def open_default_leaderboard_store() -> LeaderboardStore:
    """Resolve a store from ``KAIRU_LEADERBOARD_DB``; in-memory if unset."""
    path = os.environ.get("KAIRU_LEADERBOARD_DB", ":memory:")
    return LeaderboardStore(path)


__all__ = [
    "LeaderboardEntry",
    "LeaderboardRow",
    "LeaderboardStore",
    "open_default_leaderboard_store",
]
