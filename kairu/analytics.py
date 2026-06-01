"""Score-distribution analytics — proper server-side histogram + percentiles.

The demo's Analytics tab previously derived its histogram, percentile
bands, and >2σ anomalies client-side from a raw ``/audit`` listing.
That works for ≤500 rows but doesn't scale, doesn't compose with other
clients (notebooks, dashboards), and forces every caller to redo the
same arithmetic.

This module owns the canonical implementation. It can run against any
sequence of (id, score) rows — typically pulled from the audit log or
the leaderboard table — and returns a single ``DistributionReport``
that the API hands back as JSON.

Statistics
----------
* **Histogram**: ``n_bins`` equal-width buckets over ``[0, 1]``. Default
  20 bins matches Prometheus' default latency buckets and the demo UI.
* **Percentiles**: p5, p25, p50 (median), p75, p95 via nearest-rank
  ordering. We deliberately do *not* interpolate — for small N the
  nearest-rank percentile is the more honest summary and matches
  ``numpy.percentile`` with ``method="nearest"``.
* **Anomalies**: any row whose z-score exceeds ``anomaly_threshold``
  (default 2.0). Returned as a list of ids, sorted by |z| desc.

The implementation is pure ``statistics`` — no numpy. The hot path is
O(n log n) for the sort done once.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_N_BINS: int = 20
DEFAULT_ANOMALY_THRESHOLD: float = 2.0


@dataclass(frozen=True)
class HistogramBucket:
    """One bucket of the histogram."""

    low: float
    high: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"low": self.low, "high": self.high, "count": self.count}


@dataclass(frozen=True)
class AnomalousRow:
    """One row flagged as anomalous by the z-score test."""

    id: Any
    score: float
    z_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "score": self.score, "z_score": self.z_score}


@dataclass(frozen=True)
class DistributionReport:
    """Top-level analytics payload."""

    n: int
    mean: float
    stdev: float
    minimum: float
    maximum: float
    percentiles: Dict[str, float]   # p5, p25, p50, p75, p95
    histogram: Tuple[HistogramBucket, ...]
    bin_width: float
    anomalies: Tuple[AnomalousRow, ...]
    anomaly_threshold: float
    filters: Dict[str, Any] = field(default_factory=dict)  # echoes the request scope

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
            "percentiles": dict(self.percentiles),
            "histogram": [b.to_dict() for b in self.histogram],
            "bin_width": self.bin_width,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "anomaly_threshold": self.anomaly_threshold,
            "filters": dict(self.filters),
        }


# ─────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────


def _nearest_rank_percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (NIST method, no interpolation).

    For empty input returns 0.0; for p ∈ [0, 1] returns the element at
    rank ``max(1, ceil(p * n)) - 1`` of the sorted list (0-indexed).
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    idx = int(round(p * n + 0.5)) - 1
    idx = max(0, min(n - 1, idx))
    return float(sorted_values[idx])


def _build_histogram(values: Sequence[float], n_bins: int) -> Tuple[List[HistogramBucket], float]:
    """20 equal-width buckets over [0, 1] by default.

    Values exactly equal to 1.0 fall into the last bucket (closed-right
    at the top edge).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    width = 1.0 / n_bins
    counts = [0] * n_bins
    for v in values:
        if v < 0.0 or v > 1.0:
            # Clamp silently — scores are bounded but rounding can spill.
            v = max(0.0, min(1.0, v))
        idx = int(v / width)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1
    buckets = [
        HistogramBucket(low=i * width, high=(i + 1) * width, count=c)
        for i, c in enumerate(counts)
    ]
    return buckets, width


# ─────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────


def compute_distribution(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str = "aggregate",
    n_bins: int = DEFAULT_N_BINS,
    anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    filters: Optional[Mapping[str, Any]] = None,
) -> DistributionReport:
    """Compute the full distribution report for ``rows``.

    Each row must carry at least:

    * an ``id`` (any hashable type — used to surface anomaly back-pointers)
    * one of:
        - ``aggregate`` (numeric)  when ``metric == "aggregate"``
        - ``criteria[metric]``     when ``metric`` is a criterion name
        - ``scores[metric]``       (alternative key, matches audit-log rows)
    """
    if anomaly_threshold < 0:
        raise ValueError("anomaly_threshold must be >= 0")

    extracted: List[Tuple[Any, float]] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        if metric == "aggregate":
            v = r.get("aggregate")
            if v is None and "scores" in r and isinstance(r["scores"], Mapping):
                vals = [x for x in r["scores"].values() if isinstance(x, (int, float))]
                v = sum(vals) / len(vals) if vals else None
        else:
            v = None
            for key in ("criteria", "scores"):
                bucket = r.get(key)
                if isinstance(bucket, Mapping) and metric in bucket:
                    v = bucket[metric]
                    break
        if not isinstance(v, (int, float)):
            continue
        extracted.append((r.get("id"), float(v)))

    n = len(extracted)
    if n == 0:
        return DistributionReport(
            n=0, mean=0.0, stdev=0.0, minimum=0.0, maximum=0.0,
            percentiles={"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
            histogram=tuple(HistogramBucket(low=i / n_bins, high=(i + 1) / n_bins, count=0)
                            for i in range(n_bins)),
            bin_width=1.0 / n_bins,
            anomalies=(),
            anomaly_threshold=anomaly_threshold,
            filters=dict(filters or {}),
        )

    values = [v for _, v in extracted]
    sorted_vals = sorted(values)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if n > 1 else 0.0

    pcts = {
        "p5":  _nearest_rank_percentile(sorted_vals, 0.05),
        "p25": _nearest_rank_percentile(sorted_vals, 0.25),
        "p50": _nearest_rank_percentile(sorted_vals, 0.50),
        "p75": _nearest_rank_percentile(sorted_vals, 0.75),
        "p95": _nearest_rank_percentile(sorted_vals, 0.95),
    }

    buckets, width = _build_histogram(values, n_bins)

    anomalies: List[AnomalousRow] = []
    if stdev > 0:
        for row_id, v in extracted:
            z = (v - mean) / stdev
            if abs(z) >= anomaly_threshold:
                anomalies.append(AnomalousRow(id=row_id, score=v, z_score=z))
        anomalies.sort(key=lambda a: -abs(a.z_score))

    return DistributionReport(
        n=n,
        mean=mean,
        stdev=stdev,
        minimum=sorted_vals[0],
        maximum=sorted_vals[-1],
        percentiles=pcts,
        histogram=tuple(buckets),
        bin_width=width,
        anomalies=tuple(anomalies),
        anomaly_threshold=anomaly_threshold,
        filters=dict(filters or {}),
    )


__all__ = [
    "AnomalousRow",
    "DistributionReport",
    "HistogramBucket",
    "compute_distribution",
    "DEFAULT_N_BINS",
    "DEFAULT_ANOMALY_THRESHOLD",
]
