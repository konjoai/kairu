"""Tests for kairu.metrics_export — Prometheus exposition format."""
from __future__ import annotations

import re

import pytest

from kairu.metrics_export import (
    CONTENT_TYPE,
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
)


def test_counter_inc_and_render():
    c = Counter("test_total", "help text")
    c.inc(endpoint="/a")
    c.inc(2.0, endpoint="/a")
    c.inc(endpoint="/b")
    out = "\n".join(c.render())
    assert "# HELP test_total help text" in out
    assert "# TYPE test_total counter" in out
    assert 'test_total{endpoint="/a"} 3.0' in out
    assert 'test_total{endpoint="/b"} 1.0' in out


def test_counter_rejects_negative():
    c = Counter("x", "x")
    with pytest.raises(ValueError):
        c.inc(-1.0)


def test_gauge_set_inc_dec():
    g = Gauge("active", "in flight")
    g.set(5.0)
    g.inc()
    g.dec(2.0)
    out = "\n".join(g.render())
    assert "active 4.0" in out


def test_histogram_buckets_and_count():
    h = Histogram("latency", "h", buckets=(0.1, 0.5, 1.0))
    for v in (0.05, 0.2, 0.6, 2.0):
        h.observe(v)
    out = "\n".join(h.render())
    # 0.05 → all 4 buckets +Inf hit cumulatively. Specifically:
    #   le=0.1: 1 (only 0.05)
    #   le=0.5: 2 (0.05, 0.2)
    #   le=1.0: 3 (0.05, 0.2, 0.6)
    #   le=+Inf: 4
    assert 'latency_bucket{le="0.1"} 1' in out
    assert 'latency_bucket{le="0.5"} 2' in out
    assert 'latency_bucket{le="1.0"} 3' in out
    assert 'latency_bucket{le="+Inf"} 4' in out
    assert "latency_sum 2.85" in out
    assert "latency_count 4" in out


def test_histogram_rejects_unsorted_buckets():
    with pytest.raises(ValueError):
        Histogram("x", "x", buckets=(1.0, 0.5))


def test_collector_renders_all_series():
    m = MetricsCollector()
    m.requests_total.inc(endpoint="/health", status="200")
    m.tokens_generated_total.inc(42, finish_reason="length")
    m.errors_total.inc(kind="validation")
    m.rate_limited_total.inc()
    m.active_streams.inc()
    m.request_duration_seconds.observe(0.123, endpoint="/generate")
    m.token_latency_seconds.observe(0.005)
    out = m.render()
    assert "# TYPE kairu_requests_total counter" in out
    assert "# TYPE kairu_tokens_generated_total counter" in out
    assert "# TYPE kairu_errors_total counter" in out
    assert "# TYPE kairu_rate_limited_total counter" in out
    assert "# TYPE kairu_active_streams gauge" in out
    assert "# TYPE kairu_request_duration_seconds histogram" in out
    assert "# TYPE kairu_token_latency_seconds histogram" in out
    assert "# TYPE kairu_process_uptime_seconds gauge" in out
    # Trailing newline
    assert out.endswith("\n")


def test_content_type_is_prometheus_text():
    assert "text/plain" in CONTENT_TYPE
    assert "version=0.0.4" in CONTENT_TYPE


def test_label_escaping_in_render():
    c = Counter("x_total", "h")
    c.inc(endpoint='weird"path')
    out = "\n".join(c.render())
    assert 'endpoint="weird\\"path"' in out


def test_collector_uptime_is_monotonic_nonnegative():
    m = MetricsCollector()
    out = m.render()
    match = re.search(r"kairu_process_uptime_seconds (\d+(\.\d+)?)", out)
    assert match
    assert float(match.group(1)) >= 0.0
