"""Tests for kairu.benchmarks — reference percentile distributions."""
from __future__ import annotations

import pytest

from kairu.benchmarks import (
    BENCHMARK_CORPUS_SIZE,
    BENCHMARKS,
    HISTOGRAM_BUCKETS,
    BenchmarkStats,
    percentile_rank,
)
from kairu.evaluation import CRITERIA


def test_benchmarks_cover_every_criterion():
    assert set(BENCHMARKS.keys()) == set(CRITERIA.keys())


def test_corpus_size_is_one_thousand():
    for stats in BENCHMARKS.values():
        assert stats.n == BENCHMARK_CORPUS_SIZE


def test_quantiles_ordered():
    for name, stats in BENCHMARKS.items():
        assert 0.0 <= stats.p25 <= stats.p50 <= stats.p75 <= stats.p90 <= stats.p99 <= 1.0, name


def test_histogram_buckets_sum_to_n():
    for name, stats in BENCHMARKS.items():
        assert len(stats.histogram) == HISTOGRAM_BUCKETS
        assert sum(stats.histogram) == stats.n, name


def test_stats_dict_shape():
    name = next(iter(BENCHMARKS))
    out = BENCHMARKS[name].to_dict()
    for k in ("criterion", "n", "mean", "stdev", "p25", "p50", "p75", "p90", "p99",
              "histogram", "buckets", "samples_hash"):
        assert k in out


def test_percentile_rank_bounds():
    name = next(iter(BENCHMARKS))
    assert percentile_rank(name, -0.5) == 0.0
    assert percentile_rank(name, 0.0) == 0.0
    assert percentile_rank(name, 1.0) == 1.0
    assert percentile_rank(name, 1.5) == 1.0


def test_percentile_rank_monotone():
    name = next(iter(BENCHMARKS))
    last = -1.0
    for v in (0.1, 0.25, 0.4, 0.6, 0.8, 0.95):
        r = percentile_rank(name, v)
        assert 0.0 <= r <= 1.0
        assert r >= last - 1e-9
        last = r


def test_percentile_rank_unknown_criterion_raises():
    with pytest.raises(KeyError):
        percentile_rank("does-not-exist", 0.5)


def test_deterministic_across_imports():
    """Re-importing should yield the same samples_hash for each criterion —
    the corpus + scorer must be stable across processes."""
    first = {k: v.samples_hash for k, v in BENCHMARKS.items()}
    import importlib
    import kairu.benchmarks as m
    importlib.reload(m)
    second = {k: v.samples_hash for k, v in m.BENCHMARKS.items()}
    assert first == second


def test_stats_immutable():
    name = next(iter(BENCHMARKS))
    stats = BENCHMARKS[name]
    with pytest.raises((AttributeError, TypeError)):
        stats.p50 = 0.0  # type: ignore[misc]


def test_histogram_is_tuple_of_ints():
    for stats in BENCHMARKS.values():
        assert isinstance(stats.histogram, tuple)
        assert all(isinstance(c, int) for c in stats.histogram)
        assert all(c >= 0 for c in stats.histogram)
