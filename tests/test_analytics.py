"""Tests for kairu.analytics — score-distribution analytics."""
from __future__ import annotations

import json

import pytest

from kairu.analytics import (
    DEFAULT_ANOMALY_THRESHOLD,
    DistributionReport,
    HistogramBucket,
    compute_distribution,
)


def _rows(scores, ids=None):
    out = []
    for i, s in enumerate(scores):
        out.append({"id": (ids[i] if ids else i), "aggregate": s})
    return out


def test_empty_input_returns_zero_report():
    r = compute_distribution([])
    assert r.n == 0
    assert r.mean == 0.0
    assert r.stdev == 0.0
    assert r.anomalies == ()
    assert sum(b.count for b in r.histogram) == 0


def test_uniform_distribution_has_flat_histogram():
    scores = [i / 99.0 for i in range(100)]
    r = compute_distribution(_rows(scores), n_bins=10)
    # Every bucket should have ~10 entries.
    for b in r.histogram:
        assert 8 <= b.count <= 12


def test_percentiles_match_nearest_rank():
    # 0.0, 0.1, ..., 1.0 — 11 values. Nearest-rank with our rounding:
    # p=0.25 → idx 2 → 0.2; p=0.50 → idx 5 → 0.5; p=0.75 → idx 8 → 0.8.
    scores = [i / 10.0 for i in range(11)]
    r = compute_distribution(_rows(scores))
    assert r.percentiles["p25"] == pytest.approx(0.2, abs=1e-9)
    assert r.percentiles["p50"] == pytest.approx(0.5, abs=1e-9)
    assert r.percentiles["p75"] == pytest.approx(0.8, abs=1e-9)
    # And percentiles are monotonically ordered.
    assert r.percentiles["p5"] <= r.percentiles["p25"]
    assert r.percentiles["p75"] <= r.percentiles["p95"]


def test_percentile_extremes():
    scores = [0.1, 0.5, 0.9]
    r = compute_distribution(_rows(scores))
    assert r.percentiles["p5"] == pytest.approx(0.1, abs=1e-9)
    assert r.percentiles["p95"] == pytest.approx(0.9, abs=1e-9)


def test_anomalies_detected_above_threshold():
    scores = [0.5] * 20 + [0.99, 0.01]  # two clear outliers
    rows = _rows(scores)
    r = compute_distribution(rows, anomaly_threshold=1.0)
    flagged_ids = {a.id for a in r.anomalies}
    assert 20 in flagged_ids  # 0.99
    assert 21 in flagged_ids  # 0.01


def test_no_anomalies_when_stdev_zero():
    scores = [0.5] * 10
    r = compute_distribution(_rows(scores))
    assert r.anomalies == ()
    assert r.stdev == 0.0


def test_histogram_bucket_widths_sum_to_one():
    r = compute_distribution(_rows([0.5]), n_bins=20)
    width_sum = sum(b.high - b.low for b in r.histogram)
    assert width_sum == pytest.approx(1.0, abs=1e-9)


def test_score_at_upper_edge_falls_into_last_bucket():
    r = compute_distribution(_rows([1.0]), n_bins=5)
    counts = [b.count for b in r.histogram]
    assert counts[-1] == 1
    assert sum(counts[:-1]) == 0


def test_criterion_metric_uses_nested_field():
    rows = [
        {"id": 1, "aggregate": 0.5, "criteria": {"relevance": 0.9}},
        {"id": 2, "aggregate": 0.7, "criteria": {"relevance": 0.2}},
        # Row missing the criterion — should be silently excluded.
        {"id": 3, "aggregate": 0.6},
    ]
    r = compute_distribution(rows, metric="relevance")
    assert r.n == 2


def test_scores_dict_supplies_criterion_when_criteria_absent():
    rows = [
        {"id": 1, "scores": {"relevance": 0.4}},
        {"id": 2, "scores": {"relevance": 0.8}},
    ]
    r = compute_distribution(rows, metric="relevance")
    assert r.n == 2
    assert r.mean == pytest.approx(0.6, abs=1e-9)


def test_aggregate_derived_from_scores_when_absent():
    rows = [{"id": 1, "scores": {"a": 0.4, "b": 0.6}}]  # mean = 0.5
    r = compute_distribution(rows, metric="aggregate")
    assert r.n == 1
    assert r.mean == pytest.approx(0.5, abs=1e-9)


def test_non_mapping_rows_silently_skipped():
    rows = [{"id": 1, "aggregate": 0.5}, "not a dict", None, {"id": 2, "aggregate": 0.7}]
    r = compute_distribution(rows)
    assert r.n == 2


def test_clamps_out_of_range_values_into_histogram():
    # Negative input should land in bin 0; >1 should land in last bin.
    rows = [{"id": 0, "aggregate": -0.5}, {"id": 1, "aggregate": 1.5}]
    r = compute_distribution(rows, n_bins=4)
    assert r.histogram[0].count == 1
    assert r.histogram[-1].count == 1


def test_rejects_negative_anomaly_threshold():
    with pytest.raises(ValueError):
        compute_distribution([], anomaly_threshold=-0.1)


def test_rejects_too_few_bins():
    with pytest.raises(ValueError):
        compute_distribution([{"id": 1, "aggregate": 0.5}], n_bins=0)


def test_to_dict_is_json_serialisable():
    rows = [{"id": i, "aggregate": i / 9} for i in range(10)]
    r = compute_distribution(rows, anomaly_threshold=1.5,
                              filters={"days": 30, "model": "alpha"})
    encoded = json.dumps(r.to_dict())
    decoded = json.loads(encoded)
    assert "histogram" in decoded
    assert decoded["filters"]["model"] == "alpha"


def test_anomalies_sorted_by_absolute_z():
    rows = [{"id": i, "aggregate": 0.5} for i in range(10)]
    rows[0] = {"id": "extreme_high", "aggregate": 0.99}
    rows[1] = {"id": "mild_low",     "aggregate": 0.30}
    r = compute_distribution(rows, anomaly_threshold=0.5)
    if len(r.anomalies) >= 2:
        # First anomaly should have larger |z| than the second.
        assert abs(r.anomalies[0].z_score) >= abs(r.anomalies[1].z_score)
