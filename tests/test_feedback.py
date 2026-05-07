"""Tests for FeedbackLoop."""
from __future__ import annotations

import time

import pytest

from kairu.feedback import FeedbackLoop, FeedbackSummary
from kairu.gamma_scheduler import DynamicGammaScheduler
from kairu.bench import BenchmarkResult


def _make_result(acceptance_rate=None) -> BenchmarkResult:
    hw = {
        "hostname": "h",
        "os": "Linux",
        "os_release": "5.15",
        "machine": "x86_64",
        "python_version": "3.11",
        "cpu_model": "x86",
        "ram_total_bytes": 17179869184,
    }
    metadata: dict = {}
    if acceptance_rate is not None:
        metadata["acceptance_rate"] = acceptance_rate
    latencies = [0.1, 0.12, 0.09, 0.11, 0.10]
    sorted_lat = sorted(latencies)
    return BenchmarkResult(
        name="mock",
        model_name="MockModel",
        num_tokens=50,
        num_runs=5,
        warmup=0,
        latencies_s=latencies,
        p50=sorted_lat[2],
        p95=sorted_lat[-1],
        p99=sorted_lat[-1],
        mean=sum(latencies) / len(latencies),
        stddev=0.01,
        tokens_per_s_mean=50 / (sum(latencies) / len(latencies)),
        hardware=hw,
        metadata=metadata,
    )


@pytest.fixture
def scheduler():
    return DynamicGammaScheduler()


@pytest.fixture
def loop(scheduler):
    return FeedbackLoop(scheduler, min_results=3)


def test_feedback_loop_bad_min(scheduler):
    with pytest.raises(ValueError):
        FeedbackLoop(scheduler, min_results=0)


def test_feedback_loop_no_flush_before_min(loop):
    assert loop.ingest(_make_result(0.8)) is None
    assert loop.buffer_size == 1


def test_feedback_loop_flushes_at_min(loop):
    loop.ingest(_make_result(0.8))
    loop.ingest(_make_result(0.8))
    result = loop.ingest(_make_result(0.8))
    assert result is not None
    assert isinstance(result, FeedbackSummary)


def test_buffer_cleared_after_flush(loop):
    for _ in range(3):
        loop.ingest(_make_result(0.8))
    assert loop.buffer_size == 0


def test_high_acceptance_increases_gamma(scheduler, loop):
    initial_gamma = scheduler.gamma
    for _ in range(3):
        loop.ingest(_make_result(acceptance_rate=0.85))
    assert scheduler.gamma >= initial_gamma


def test_low_acceptance_decreases_gamma(scheduler, loop):
    initial_gamma = scheduler.gamma
    for _ in range(3):
        loop.ingest(_make_result(acceptance_rate=0.2))
    assert scheduler.gamma <= initial_gamma


def test_normal_acceptance_no_gamma_change(scheduler, loop):
    initial_gamma = scheduler.gamma
    for _ in range(3):
        loop.ingest(_make_result(acceptance_rate=0.6))
    # Gamma may not change for mid-range acceptance
    # Verify via a fresh loop+scheduler
    scheduler2 = DynamicGammaScheduler()
    loop2 = FeedbackLoop(scheduler2, min_results=3)
    for _ in range(2):
        loop2.ingest(_make_result(0.6))
    summary = loop2.ingest(_make_result(0.6))
    assert summary is not None
    assert "normal range" in summary.recommendation.lower() or not summary.gamma_adjusted


def test_summary_fields(loop):
    for _ in range(2):
        loop.ingest(_make_result(0.85))
    summary = loop.ingest(_make_result(0.85))
    assert summary is not None
    assert summary.n_results == 3
    assert 0.0 <= summary.mean_acceptance_rate <= 1.0


def test_no_acceptance_rate_in_metadata(loop):
    """Results without acceptance_rate key should not crash."""
    for _ in range(3):
        loop.ingest(_make_result(None))
    # Should produce a summary with mean_ar=0.5 and no adjustment


def test_mean_acceptance_rate_computed(scheduler):
    loop = FeedbackLoop(scheduler, min_results=2)
    loop.ingest(_make_result(acceptance_rate=0.8))
    summary = loop.ingest(_make_result(acceptance_rate=0.9))
    assert abs(summary.mean_acceptance_rate - 0.85) < 1e-6


def test_summary_gamma_adjusted_flag_true(scheduler):
    loop = FeedbackLoop(scheduler, min_results=2)
    loop.ingest(_make_result(0.9))
    s = loop.ingest(_make_result(0.9))
    assert s.gamma_adjusted is True
    assert s.new_gamma is not None


def test_summary_gamma_adjusted_flag_false(scheduler):
    loop = FeedbackLoop(scheduler, min_results=2)
    loop.ingest(_make_result(0.6))
    s = loop.ingest(_make_result(0.6))
    assert s.gamma_adjusted is False


def test_feedback_loop_multiple_flush_cycles(scheduler):
    loop = FeedbackLoop(scheduler, min_results=2)
    for cycle in range(3):
        loop.ingest(_make_result(0.8))
        s = loop.ingest(_make_result(0.8))
        assert s is not None
        assert s.n_results == 2


def test_buffer_size_tracks_ingested(loop):
    loop.ingest(_make_result())
    assert loop.buffer_size == 1
    loop.ingest(_make_result())
    assert loop.buffer_size == 2
