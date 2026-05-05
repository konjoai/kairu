"""Tests for benchmarks/run_corpus.py — CorpusBenchmarkRunner, CLI, corpus shape.

All tests run fully offline using MockModel. No HF models required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.run_corpus import (
    CORPUS,
    CorpusBenchmarkRunner,
    build_parser,
    main,
)
from kairu.mock_model import MockModel


# ---------------------------------------------------------------------------
# Test 1 — CORPUS has exactly 100 prompts
# ---------------------------------------------------------------------------

def test_corpus_length() -> None:
    assert len(CORPUS) == 100, f"Expected 100 prompts, got {len(CORPUS)}"


# ---------------------------------------------------------------------------
# Test 2 — All corpus prompts are non-empty strings
# ---------------------------------------------------------------------------

def test_corpus_prompts_are_nonempty_strings() -> None:
    for i, p in enumerate(CORPUS):
        assert isinstance(p, str), f"Prompt {i} is not a string"
        assert len(p.strip()) > 0, f"Prompt {i} is blank"


# ---------------------------------------------------------------------------
# Test 3 — CorpusBenchmarkRunner produces correct shape
# ---------------------------------------------------------------------------

def test_corpus_runner_shape() -> None:
    runner = CorpusBenchmarkRunner(MockModel(), name="test-corpus")
    # Use a tiny sub-corpus of 5 prompts for speed.
    result = runner.run(num_tokens=10, warmup=1, prompts=CORPUS[:5])
    assert result.name == "test-corpus"
    assert result.model_name == "MockModel"
    assert result.num_tokens == 10
    assert result.num_runs == 5
    assert result.warmup == 1
    assert len(result.latencies_s) == 5
    assert all(lat > 0 for lat in result.latencies_s)


# ---------------------------------------------------------------------------
# Test 4 — p50 <= p95 <= p99 ordering
# ---------------------------------------------------------------------------

def test_percentile_ordering() -> None:
    runner = CorpusBenchmarkRunner(MockModel(), name="ordering")
    result = runner.run(num_tokens=10, warmup=1, prompts=CORPUS[:10])
    assert result.p50 <= result.p95, f"p50={result.p50} > p95={result.p95}"
    assert result.p95 <= result.p99, f"p95={result.p95} > p99={result.p99}"


# ---------------------------------------------------------------------------
# Test 5 — tokens_per_s_mean is positive
# ---------------------------------------------------------------------------

def test_tokens_per_s_positive() -> None:
    runner = CorpusBenchmarkRunner(MockModel(), name="tps")
    result = runner.run(num_tokens=10, warmup=0, prompts=CORPUS[:5])
    assert result.tokens_per_s_mean > 0


# ---------------------------------------------------------------------------
# Test 6 — to_json() round-trip contains required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "name", "model_name", "num_tokens", "num_runs", "warmup",
    "latencies_s", "p50", "p95", "p99", "mean", "stddev",
    "tokens_per_s_mean", "hardware", "timestamp",
}


def test_to_json_round_trip() -> None:
    runner = CorpusBenchmarkRunner(MockModel(), name="json-rt")
    result = runner.run(num_tokens=5, warmup=0, prompts=CORPUS[:3])
    parsed = json.loads(result.to_json())
    missing = REQUIRED_KEYS - set(parsed.keys())
    assert not missing, f"Missing JSON keys: {missing}"


# ---------------------------------------------------------------------------
# Test 7 — save() writes valid JSON and never overwrites
# ---------------------------------------------------------------------------

def test_save_no_overwrite(tmp_path) -> None:
    runner = CorpusBenchmarkRunner(MockModel(), name="save-test")
    r1 = runner.run(num_tokens=5, warmup=0, prompts=CORPUS[:3])
    # Force the same timestamp so we can check the suffix logic.
    r1.timestamp = "20260101T000000Z"
    r2 = runner.run(num_tokens=5, warmup=0, prompts=CORPUS[:3])
    r2.timestamp = "20260101T000000Z"
    p1 = r1.save(base_dir=str(tmp_path))
    p2 = r2.save(base_dir=str(tmp_path))
    assert p1.exists()
    assert p2.exists()
    assert p1 != p2, "save() must not overwrite an existing file"


# ---------------------------------------------------------------------------
# Test 8 — CLI main() exits 0 with mock model and tiny corpus
# ---------------------------------------------------------------------------

def test_cli_main_exits_0(tmp_path) -> None:
    exit_code = main([
        "--model", "mock",
        "--tokens", "5",
        "--warmup", "1",
        "--runs", "5",
        "--name", "ci-corpus",
        "--output", str(tmp_path),
    ])
    assert exit_code == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1, "CLI must save exactly one result file"


# ---------------------------------------------------------------------------
# Test 9 — CLI --help exits 0
# ---------------------------------------------------------------------------

def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Test 10 — Hardware metadata is collected and non-empty
# ---------------------------------------------------------------------------

def test_hardware_metadata_present() -> None:
    runner = CorpusBenchmarkRunner(MockModel(), name="hw-test")
    result = runner.run(num_tokens=5, warmup=0, prompts=CORPUS[:3])
    assert isinstance(result.hardware, dict)
    assert "hostname" in result.hardware
    assert "python_version" in result.hardware
