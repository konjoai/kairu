"""SPEED-Bench-style semantic task splits for benchmarking.

Speculative-decoding speedup and early-exit savings are strongly
*task-dependent*: a model drafts structured code or repetitive text far more
predictably than open-ended dialogue, so a single aggregate latency number
hides the variance that actually matters at deploy time. Following the
SPEED-Bench methodology, this module runs the benchmark across a set of named
semantic task splits and reports per-split throughput plus a dispersion measure
(coefficient of variation) that quantifies exactly how task-sensitive the
configuration is.

Pure stdlib + :mod:`kairu.bench` — no ML frameworks. Each split carries a
deterministic integer prompt so :class:`~kairu.mock_model.MockModel` runs fully
offline; real tokenised prompts plug in behind the same :class:`TaskSplit`
contract.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from kairu.base import ModelInterface
from kairu.bench import BenchmarkRunner


@dataclass(frozen=True)
class TaskSplit:
    """A named semantic task paired with a representative prompt."""

    name: str
    prompt: list[int]
    description: str = ""


# Distinct deterministic prompts — the split *labels* carry the semantics; the
# token ids merely need to differ so each split exercises a different prefix.
DEFAULT_SPLITS: tuple[TaskSplit, ...] = (
    TaskSplit("translation", [2, 4, 8, 16, 32], "short-context sequence transduction"),
    TaskSplit("summarization", [5, 10, 15, 20, 25, 30], "long-context compression"),
    TaskSplit("qa", [7, 14, 21], "short factual question answering"),
    TaskSplit("code", [1, 1, 2, 3, 5, 8, 13], "structured, low-entropy generation"),
    TaskSplit("dialogue", [9, 18, 27, 36], "open-ended conversational turns"),
    TaskSplit("math", [3, 9, 27, 81], "step-by-step numeric reasoning"),
)


@dataclass(frozen=True)
class SplitResult:
    """Per-split latency / throughput summary."""

    name: str
    p50_s: float
    mean_s: float
    tokens_per_s: float

    def as_dict(self) -> dict:
        """JSON-friendly mapping of the split's metrics."""
        return {
            "name": self.name,
            "p50_s": self.p50_s,
            "mean_s": self.mean_s,
            "tokens_per_s": self.tokens_per_s,
        }


@dataclass(frozen=True)
class SpeedBenchReport:
    """Cross-split benchmark report quantifying task-dependence."""

    splits: tuple[SplitResult, ...]
    fastest: str
    slowest: str
    mean_tokens_per_s: float
    # Coefficient of variation of throughput across splits — 0.0 means the
    # configuration performs uniformly regardless of task; larger means more
    # task-sensitive (the speedup you measure depends heavily on the workload).
    throughput_cv: float

    def as_dict(self) -> dict:
        """JSON-friendly mapping of the full report."""
        return {
            "splits": [s.as_dict() for s in self.splits],
            "fastest": self.fastest,
            "slowest": self.slowest,
            "mean_tokens_per_s": self.mean_tokens_per_s,
            "throughput_cv": self.throughput_cv,
        }


def run_speed_bench(
    model: ModelInterface,
    splits: tuple[TaskSplit, ...] | None = None,
    *,
    num_tokens: int = 64,
    num_runs: int = 20,
    warmup: int = 3,
) -> SpeedBenchReport:
    """Benchmark ``model`` across semantic task splits.

    Drives one :class:`~kairu.bench.BenchmarkRunner` per split and returns a
    :class:`SpeedBenchReport` carrying per-split throughput, the fastest/slowest
    split by tokens/s, the cross-split mean throughput, and the throughput
    coefficient of variation — a single number capturing how task-sensitive the
    configuration is (``0.0`` → uniform across tasks).
    """
    chosen = tuple(splits) if splits is not None else DEFAULT_SPLITS
    if not chosen:
        raise ValueError("at least one task split is required")

    results: list[SplitResult] = []
    for split in chosen:
        runner = BenchmarkRunner(model, name=split.name, prompt=split.prompt)
        r = runner.run(num_tokens=num_tokens, num_runs=num_runs, warmup=warmup)
        results.append(
            SplitResult(
                name=split.name,
                p50_s=r.p50,
                mean_s=r.mean,
                tokens_per_s=r.tokens_per_s_mean,
            )
        )

    throughputs = [s.tokens_per_s for s in results]
    mean_tps = statistics.mean(throughputs)
    cv = (
        statistics.stdev(throughputs) / mean_tps
        if len(throughputs) >= 2 and mean_tps > 0
        else 0.0
    )
    fastest = max(results, key=lambda s: s.tokens_per_s).name
    slowest = min(results, key=lambda s: s.tokens_per_s).name

    return SpeedBenchReport(
        splits=tuple(results),
        fastest=fastest,
        slowest=slowest,
        mean_tokens_per_s=mean_tps,
        throughput_cv=cv,
    )


__all__ = [
    "TaskSplit",
    "SplitResult",
    "SpeedBenchReport",
    "DEFAULT_SPLITS",
    "run_speed_bench",
]
