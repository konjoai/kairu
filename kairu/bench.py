"""Benchmarking suite for kairu inference runs.

BenchmarkRunner drives N generation runs against any ModelInterface and
computes p50/p95/p99/stddev latency statistics over per-run total-token
latencies.  Results are saved to benchmarks/results/<timestamp>_<name>.json;
previous runs are NEVER overwritten.

Usage (programmatic)::

    from kairu.bench import BenchmarkRunner
    from kairu.mock_model import MockModel

    runner = BenchmarkRunner(MockModel(), name="quick-check")
    result = runner.run(num_tokens=100, num_runs=50, warmup=5)
    print(result.to_json())
    result.save()

Usage (CLI)::

    python -m kairu.bench --model mock --tokens 100 --runs 50 --warmup 5
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kairu.base import ModelInterface
from kairu.mock_model import MockModel
from kairu.streaming import StreamingDecoder


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------

def _collect_hardware() -> dict:
    """Collect host hardware metadata without requiring any ML libraries.

    Falls back gracefully at every step so this never raises on any platform.
    """
    hw: dict = {}

    hw["hostname"] = platform.node()
    hw["os"] = platform.system()
    hw["os_release"] = platform.release()
    hw["machine"] = platform.machine()
    hw["python_version"] = sys.version

    # --- CPU model ---
    cpu = "unknown"
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            cpu = out.decode().strip()
        except Exception:  # noqa: BLE001
            pass
    if cpu == "unknown":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except Exception:  # noqa: BLE001
            pass
    hw["cpu_model"] = cpu

    # --- Total RAM ---
    ram_bytes: int | None = None
    try:
        import psutil  # type: ignore[import]

        ram_bytes = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001
        pass

    if ram_bytes is None and platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            ram_bytes = int(out.decode().strip())
        except Exception:  # noqa: BLE001
            pass

    if ram_bytes is None:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # value is in kB
                        ram_bytes = int(line.split()[1]) * 1024
                        break
        except Exception:  # noqa: BLE001
            pass

    hw["ram_total_bytes"] = ram_bytes

    return hw


# ---------------------------------------------------------------------------
# Percentile helper (pure statistics / sorted-list — no scipy)
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) using linear interpolation.

    ``sorted_data`` must already be sorted in ascending order.
    """
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    # Nearest-rank with linear interpolation (same semantics as numpy p50/p95/p99)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Immutable record of a completed benchmark run."""

    name: str
    model_name: str
    num_tokens: int
    num_runs: int
    warmup: int

    # Per-run total-latency list (warmup excluded), seconds
    latencies_s: list[float]

    # Derived statistics (over latencies_s)
    p50: float
    p95: float
    p99: float
    mean: float
    stddev: float

    tokens_per_s_mean: float

    hardware: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "num_tokens": self.num_tokens,
            "num_runs": self.num_runs,
            "warmup": self.warmup,
            "latencies_s": self.latencies_s,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
            "stddev": self.stddev,
            "tokens_per_s_mean": self.tokens_per_s_mean,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string with indent=2."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, base_dir: str = "benchmarks/results") -> Path:
        """Save result JSON to ``<base_dir>/<timestamp>_<name>.json``.

        Creates parent directories as needed.  NEVER overwrites an existing
        file — appends a numeric suffix if a collision occurs (extremely rare
        given ISO-second resolution timestamps, but guaranteed safe).
        """
        out_dir = Path(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.timestamp}_{self.name}"
        candidate = out_dir / f"{stem}.json"
        counter = 1
        while candidate.exists():
            candidate = out_dir / f"{stem}_{counter}.json"
            counter += 1

        candidate.write_text(self.to_json(), encoding="utf-8")
        return candidate


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Drive N generation runs against a ``ModelInterface`` and collect latency stats.

    Args:
        model: Any ``ModelInterface`` implementation.
        name:  Human-readable label embedded in result filenames and JSON.

    Example::

        runner = BenchmarkRunner(MockModel(), name="mock-int8")
        result = runner.run(num_tokens=100, num_runs=50, warmup=5)
        result.save()
    """

    # A fixed prompt so every run starts from the same context.
    _PROMPT: list[int] = [1, 2, 3, 4, 5]

    def __init__(self, model: ModelInterface, name: str = "benchmark") -> None:
        self._model = model
        self.name = name

    def run(
        self,
        num_tokens: int = 100,
        num_runs: int = 50,
        warmup: int = 5,
    ) -> BenchmarkResult:
        """Execute the benchmark and return a :class:`BenchmarkResult`.

        Args:
            num_tokens: Number of tokens to generate per run.
            num_runs:   Number of measured runs *after* warmup.
            warmup:     Number of warmup runs to discard.

        Returns:
            :class:`BenchmarkResult` with p50/p95/p99/stddev/mean computed
            over per-run total-token latencies in seconds.
        """
        decoder = StreamingDecoder(self._model, temperature=1.0)
        total_runs = warmup + num_runs

        raw_latencies: list[float] = []

        for i in range(total_runs):
            t0 = time.perf_counter()
            # Consume the full stream — each generated token is timed as part of
            # the per-run total latency.
            for _ in decoder.stream(self._PROMPT, max_new_tokens=num_tokens):
                pass
            elapsed = time.perf_counter() - t0

            if i >= warmup:
                raw_latencies.append(elapsed)

        # Compute statistics ---------------------------------------------------
        sorted_lat = sorted(raw_latencies)
        p50 = _percentile(sorted_lat, 50.0)
        p95 = _percentile(sorted_lat, 95.0)
        p99 = _percentile(sorted_lat, 99.0)
        mean = statistics.mean(raw_latencies) if raw_latencies else 0.0
        stddev = statistics.stdev(raw_latencies) if len(raw_latencies) >= 2 else 0.0
        tok_s_mean = (num_tokens / mean) if mean > 0 else 0.0

        model_name = type(self._model).__name__

        return BenchmarkResult(
            name=self.name,
            model_name=model_name,
            num_tokens=num_tokens,
            num_runs=num_runs,
            warmup=warmup,
            latencies_s=raw_latencies,
            p50=p50,
            p95=p95,
            p99=p99,
            mean=mean,
            stddev=stddev,
            tokens_per_s_mean=tok_s_mean,
            hardware=_collect_hardware(),
        )


# ---------------------------------------------------------------------------
# CLI helpers (also importable from kairu.__main__bench for tests)
# ---------------------------------------------------------------------------

import argparse  # noqa: E402 — deferred to keep module-level imports minimal


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m kairu.bench",
        description=(
            "Kairu benchmarking suite — measure p50/p95/p99 "
            "token-generation latency."
        ),
    )
    parser.add_argument(
        "--model",
        default="mock",
        metavar="MODEL",
        help="'mock' (no ML deps) or a HuggingFace model name (requires kairu[hf]).",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
        metavar="N",
        help="Number of tokens to generate per run (default: 100).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        metavar="N",
        help="Number of measured runs after warmup (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        metavar="N",
        help="Number of warmup runs to discard (default: 5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Override save directory (default: benchmarks/results).",
    )
    parser.add_argument(
        "--name",
        default="benchmark",
        metavar="NAME",
        help="Label embedded in the result filename and JSON (default: benchmark).",
    )
    return parser


def _load_model(model_arg: str) -> ModelInterface:
    """Instantiate the requested model backend."""
    if model_arg == "mock":
        return MockModel()

    # HuggingFace model — requires kairu[hf]
    try:
        from kairu._hf_backend import HuggingFaceModel
    except ImportError as exc:
        print(
            f"ERROR: HuggingFace backend not available.  "
            f"Install with: pip install 'kairu[hf]'\n  ({exc})",
            file=sys.stderr,
        )
        sys.exit(1)
    return HuggingFaceModel(model_arg)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* and run the benchmark.  Returns exit code (0 on success).

    Designed to be called directly from tests::

        from kairu.bench import main
        assert main(["--model", "mock", "--tokens", "10", "--runs", "3"]) == 0
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    model = _load_model(args.model)
    runner = BenchmarkRunner(model, name=args.name)

    print(
        f"\nKairu Benchmark — model={args.model!r}  "
        f"tokens={args.tokens}  runs={args.runs}  warmup={args.warmup}\n"
        f"{'─' * 56}"
    )

    result = runner.run(
        num_tokens=args.tokens,
        num_runs=args.runs,
        warmup=args.warmup,
    )

    col_w = 20
    print(f"{'Metric':<{col_w}}  {'Value':>15}")
    print(f"{'─' * col_w}  {'─' * 15}")
    print(f"{'p50 latency (s)':<{col_w}}  {result.p50:>15.6f}")
    print(f"{'p95 latency (s)':<{col_w}}  {result.p95:>15.6f}")
    print(f"{'p99 latency (s)':<{col_w}}  {result.p99:>15.6f}")
    print(f"{'mean latency (s)':<{col_w}}  {result.mean:>15.6f}")
    print(f"{'stddev (s)':<{col_w}}  {result.stddev:>15.6f}")
    print(f"{'tokens/s (mean)':<{col_w}}  {result.tokens_per_s_mean:>15.2f}")
    print(f"{'─' * col_w}  {'─' * 15}")

    save_kwargs: dict = {}
    if args.output:
        save_kwargs["base_dir"] = args.output

    saved_path = result.save(**save_kwargs)
    print(f"\nResult saved → {saved_path}\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
