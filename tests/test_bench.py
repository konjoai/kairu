"""Tests for kairu.bench — BenchmarkRunner, BenchmarkResult, CLI.

8 tests covering:
1.  BenchmarkRunner shape (runs=3, warmup=1, tokens=10).
2.  p50 <= p95 <= p99 ordering.
3.  to_json() round-trips through json.loads with all required keys.
4.  save() writes a valid JSON file with non-empty hardware section.
5.  _collect_hardware() returns dict with required keys.
6.  CLI main(["--model", "mock", "--tokens", "10", "--runs", "3"]) exits 0.
7.  CLI --help prints usage without error.
8.  Saved result path contains timestamp and name.
"""
from __future__ import annotations

import json
import sys

import pytest

from kairu.bench import (
    BenchmarkResult,
    BenchmarkRunner,
    _collect_hardware,
    main,
)
from kairu.mock_model import MockModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_result(tmp_path) -> BenchmarkResult:
    """Run 3 measured runs (1 warmup) generating 10 tokens each."""
    runner = BenchmarkRunner(MockModel(), name="test-bench")
    result = runner.run(num_tokens=10, num_runs=3, warmup=1)
    return result


# ---------------------------------------------------------------------------
# Test 1 — correct shape
# ---------------------------------------------------------------------------

def test_result_shape(small_result: BenchmarkResult) -> None:
    """BenchmarkResult has the right field values after a small run."""
    r = small_result
    assert r.name == "test-bench"
    assert r.model_name == "MockModel"
    assert r.num_tokens == 10
    assert r.num_runs == 3
    assert r.warmup == 1
    # Exactly num_runs latencies recorded (warmup excluded)
    assert len(r.latencies_s) == 3
    assert all(lat > 0 for lat in r.latencies_s), "All latencies must be positive"
    assert r.tokens_per_s_mean > 0
    assert r.mean > 0
    assert r.stddev >= 0
    assert r.timestamp != ""


# ---------------------------------------------------------------------------
# Test 2 — percentile ordering
# ---------------------------------------------------------------------------

def test_percentile_ordering(small_result: BenchmarkResult) -> None:
    """p50 <= p95 <= p99 must hold by definition."""
    r = small_result
    assert r.p50 <= r.p95, f"p50={r.p50} > p95={r.p95}"
    assert r.p95 <= r.p99, f"p95={r.p95} > p99={r.p99}"


# ---------------------------------------------------------------------------
# Test 3 — JSON round-trip with required keys
# ---------------------------------------------------------------------------

REQUIRED_JSON_KEYS = {
    "name",
    "model_name",
    "num_tokens",
    "num_runs",
    "warmup",
    "latencies_s",
    "p50",
    "p95",
    "p99",
    "mean",
    "stddev",
    "tokens_per_s_mean",
    "hardware",
    "timestamp",
}


def test_to_json_round_trip(small_result: BenchmarkResult) -> None:
    """to_json() must produce valid JSON containing all required keys."""
    raw = small_result.to_json()
    parsed = json.loads(raw)
    missing = REQUIRED_JSON_KEYS - set(parsed.keys())
    assert not missing, f"Missing keys in JSON output: {missing}"
    # Values must be consistent
    assert parsed["num_runs"] == small_result.num_runs
    assert parsed["p50"] == pytest.approx(small_result.p50, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 4 — save() writes valid JSON with non-empty hardware section
# ---------------------------------------------------------------------------

def test_save_writes_valid_json(tmp_path, small_result: BenchmarkResult) -> None:
    """save() must create a JSON file; hardware section must be non-empty."""
    saved = small_result.save(base_dir=str(tmp_path))
    assert saved.exists(), "save() must create the output file"
    content = json.loads(saved.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    hw = content.get("hardware", {})
    assert hw, "hardware section must be non-empty"
    assert "hostname" in hw, "'hostname' missing from hardware section"


# ---------------------------------------------------------------------------
# Test 5 — _collect_hardware() has required keys
# ---------------------------------------------------------------------------

def test_collect_hardware_keys() -> None:
    """_collect_hardware() must return a dict with at least the required keys."""
    hw = _collect_hardware()
    required = {"hostname", "os", "python_version"}
    missing = required - set(hw.keys())
    assert not missing, f"Missing hardware keys: {missing}"
    assert isinstance(hw["hostname"], str)
    assert isinstance(hw["python_version"], str)


# ---------------------------------------------------------------------------
# Test 6 — CLI exits 0
# ---------------------------------------------------------------------------

def test_cli_main_exits_0(tmp_path) -> None:
    """main() with --model mock must complete and return 0."""
    exit_code = main(
        [
            "--model", "mock",
            "--tokens", "10",
            "--runs", "3",
            "--warmup", "1",
            "--name", "ci-smoke",
            "--output", str(tmp_path),
        ]
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Test 7 — CLI --help exits 0
# ---------------------------------------------------------------------------

def test_cli_help(capsys) -> None:
    """--help must print usage and exit 0 (argparse raises SystemExit(0))."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower() or "kairu" in captured.out.lower()


# ---------------------------------------------------------------------------
# Test 8 — saved filename contains timestamp and name
# ---------------------------------------------------------------------------

def test_saved_filename_contains_timestamp_and_name(tmp_path) -> None:
    """The saved JSON filename must embed the result timestamp and name."""
    runner = BenchmarkRunner(MockModel(), name="my-bench")
    result = runner.run(num_tokens=5, num_runs=2, warmup=1)
    saved = result.save(base_dir=str(tmp_path))
    stem = saved.stem  # filename without .json
    assert result.timestamp in stem, f"Timestamp {result.timestamp!r} not in {stem!r}"
    assert "my-bench" in stem, f"Name 'my-bench' not in {stem!r}"
