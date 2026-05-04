"""Prometheus exposition-format metrics collector — pure stdlib.

We avoid the official ``prometheus_client`` dependency (it pulls protobufs
and a multiprocessing helper we do not need). The exposition format is a
plain text spec — three lines per series — and we render it from three
primitives:

  * :class:`Counter`   — monotonically increasing
  * :class:`Gauge`     — set / inc / dec
  * :class:`Histogram` — bucketed observations + ``_sum`` + ``_count``

Histogram buckets default to a Prometheus-canonical latency ladder
(0.005s … 10s) so dashboards built around standard quantile queries work
unmodified. ``_observe`` is O(log buckets) via ``bisect``.

Thread/asyncio safety: each primitive serializes mutations through a
``threading.Lock``. Reads under the lock too — exposition is rare and
correctness > microseconds here.
"""
from __future__ import annotations

import bisect
import threading
import time
from typing import Iterable


_DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


def _esc_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels_str(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{_esc_label(v)}"' for k, v in sorted(labels.items())) + "}"


class _Series:
    """One metric, possibly with multiple label combinations."""

    METRIC_TYPE = "untyped"

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._lock = threading.Lock()


class Counter(_Series):
    METRIC_TYPE = "counter"

    def __init__(self, name: str, help_text: str) -> None:
        super().__init__(name, help_text)
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def inc(self, value: float = 1.0, **labels: str) -> None:
        if value < 0:
            raise ValueError("counter must not decrease")
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        with self._lock:
            for key, val in sorted(self._values.items()):
                yield f"{self.name}{_labels_str(dict(key))} {val}"


class Gauge(_Series):
    METRIC_TYPE = "gauge"

    def __init__(self, name: str, help_text: str) -> None:
        super().__init__(name, help_text)
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, delta: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + delta

    def dec(self, delta: float = 1.0, **labels: str) -> None:
        self.inc(-delta, **labels)

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} gauge"
        with self._lock:
            for key, val in sorted(self._values.items()):
                yield f"{self.name}{_labels_str(dict(key))} {val}"


class Histogram(_Series):
    METRIC_TYPE = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text)
        if list(buckets) != sorted(set(buckets)):
            raise ValueError("buckets must be unique and ascending")
        self._buckets = tuple(buckets)
        # series-key → (counts_per_bucket+inf, sum, count)
        self._series: dict[tuple[tuple[str, str], ...], tuple[list[int], float, int]] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            counts, total, count = self._series.get(
                key, ([0] * (len(self._buckets) + 1), 0.0, 0)
            )
            # Bisect: smallest bucket >= value gets the increment, plus +Inf.
            idx = bisect.bisect_left(self._buckets, value)
            for i in range(idx, len(self._buckets) + 1):
                counts[i] += 1
            total += value
            count += 1
            self._series[key] = (counts, total, count)

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        with self._lock:
            for key, (counts, total, count) in sorted(self._series.items()):
                base_labels = dict(key)
                for b, c in zip(self._buckets, counts):
                    labels = {**base_labels, "le": _format_bound(b)}
                    yield f"{self.name}_bucket{_labels_str(labels)} {c}"
                inf_labels = {**base_labels, "le": "+Inf"}
                yield f"{self.name}_bucket{_labels_str(inf_labels)} {counts[-1]}"
                yield f"{self.name}_sum{_labels_str(base_labels)} {total}"
                yield f"{self.name}_count{_labels_str(base_labels)} {count}"


def _format_bound(b: float) -> str:
    """Prometheus convention: trailing zeros stripped, no exponent."""
    if b == int(b):
        return f"{int(b)}.0"
    return f"{b}"


# ─── registry ─────────────────────────────────────────────────────────────

class MetricsCollector:
    """Holds all server-side metrics. One instance per process.

    The named series exposed here are stable contracts — dashboards key on
    these names. Add freely; never rename without bumping a major version.
    """

    def __init__(self) -> None:
        self.requests_total = Counter(
            "kairu_requests_total",
            "Total HTTP requests received by endpoint and status.",
        )
        self.tokens_generated_total = Counter(
            "kairu_tokens_generated_total",
            "Total tokens emitted by /generate.",
        )
        self.errors_total = Counter(
            "kairu_errors_total",
            "Total errors broken down by kind.",
        )
        self.rate_limited_total = Counter(
            "kairu_rate_limited_total",
            "Total requests rejected by the rate limiter.",
        )
        self.active_streams = Gauge(
            "kairu_active_streams",
            "Number of in-flight /generate SSE streams.",
        )
        self.request_duration_seconds = Histogram(
            "kairu_request_duration_seconds",
            "End-to-end request duration by endpoint.",
        )
        self.token_latency_seconds = Histogram(
            "kairu_token_latency_seconds",
            "Per-token generation latency.",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.process_start_time = time.time()

    @property
    def _all(self) -> tuple[_Series, ...]:
        return (
            self.requests_total,
            self.tokens_generated_total,
            self.errors_total,
            self.rate_limited_total,
            self.active_streams,
            self.request_duration_seconds,
            self.token_latency_seconds,
        )

    def render(self) -> str:
        """Return the Prometheus exposition payload (text/plain; version=0.0.4)."""
        lines: list[str] = []
        for series in self._all:
            lines.extend(series.render())
        # Process uptime as a built-in gauge series.
        lines.append("# HELP kairu_process_uptime_seconds Process uptime in seconds.")
        lines.append("# TYPE kairu_process_uptime_seconds gauge")
        lines.append(f"kairu_process_uptime_seconds {time.time() - self.process_start_time}")
        return "\n".join(lines) + "\n"


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsCollector",
    "CONTENT_TYPE",
]
