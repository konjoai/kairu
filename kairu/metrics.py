from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GenerationMetrics:
    """
    Tracks per-generation performance metrics.
    All times are in milliseconds.

    Usage:
        m = GenerationMetrics(prompt_tokens=32)
        for tok in generated_stream:
            m.record_token()
        m.finish()
        print(m.to_dict())
    """

    prompt_tokens: int = 0
    generated_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0

    _start_time: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _end_time: float | None = field(default=None, init=False, repr=False)
    _token_timestamps: list[float] = field(default_factory=list, init=False, repr=False)

    def record_token(self) -> None:
        """Record the wall-clock timestamp of a newly emitted token."""
        self._token_timestamps.append(time.perf_counter())
        self.generated_tokens += 1

    def finish(self) -> None:
        """Seal the measurement window."""
        self._end_time = time.perf_counter()

    @property
    def total_time_ms(self) -> float:
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return (end - self._start_time) * 1000.0

    @property
    def tokens_per_second(self) -> float:
        if self.generated_tokens == 0 or self.total_time_ms < 1e-6:
            return 0.0
        return self.generated_tokens / (self.total_time_ms / 1000.0)

    @property
    def mean_latency_ms(self) -> float:
        if self.generated_tokens == 0:
            return 0.0
        return self.total_time_ms / self.generated_tokens

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_tokens + self.rejected_tokens
        if total == 0:
            return 0.0
        return self.accepted_tokens / total

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_time_ms": round(self.total_time_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "acceptance_rate": round(self.acceptance_rate, 4),
        }
