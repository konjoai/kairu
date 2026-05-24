"""Tests for kairu.ci_regression — baseline snapshots + regression checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kairu.ci_regression import (
    BaselineSnapshot,
    BaselineStore,
    FileBaselineStore,
    RegressionReport,
    check_against_baseline,
    snapshot_baseline,
)


# Three golden datapoints — different prompt difficulties so the rubric
# yields a non-trivial spread of scores per criterion.
GOLDEN = [
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
    },
    {
        "input": "Summarize Hamlet briefly.",
        "output": (
            "Hamlet is a tragedy by Shakespeare in which the Danish prince Hamlet "
            "investigates and avenges his father's murder by his uncle Claudius."
        ),
    },
    {
        "input": "Explain TCP congestion control to a software engineer in two sentences.",
        "output": (
            "TCP probes network capacity by additively growing its congestion window "
            "and multiplicatively halving it on packet loss; this is AIMD."
        ),
    },
]

DEGRADED = [
    {"input": GOLDEN[0]["input"], "output": "paris"},
    {"input": GOLDEN[1]["input"], "output": "a play about a sad guy"},
    {"input": GOLDEN[2]["input"], "output": "uhh networking is complicated"},
]


# ── snapshot_baseline ────────────────────────────────────────────────────


def test_snapshot_baseline_rejects_empty():
    with pytest.raises(ValueError):
        snapshot_baseline([])


def test_snapshot_baseline_returns_one_item_per_input():
    snap = snapshot_baseline(GOLDEN)
    assert isinstance(snap, BaselineSnapshot)
    assert snap.n_items == 3
    assert len(snap.items) == 3
    assert 0.0 <= snap.mean_aggregate <= 1.0
    # Distinct input hashes per item.
    hashes = [i.input_hash for i in snap.items]
    assert len(set(hashes)) == 3


def test_snapshot_baseline_rejects_bad_item_type():
    with pytest.raises(TypeError):
        snapshot_baseline([{"input": 123, "output": "x"}])


def test_snapshot_baseline_roundtrips_via_dict():
    snap = snapshot_baseline(GOLDEN, label="prod v1")
    d = snap.to_dict()
    snap2 = BaselineSnapshot.from_dict(d)
    assert snap2.snapshot_id == snap.snapshot_id
    assert snap2.n_items == snap.n_items
    assert snap2.label == "prod v1"
    assert snap2.items[0].input_hash == snap.items[0].input_hash


# ── check_against_baseline ──────────────────────────────────────────────


def test_check_passes_on_identical_inputs():
    snap = snapshot_baseline(GOLDEN)
    report = check_against_baseline(snap, GOLDEN)
    assert isinstance(report, RegressionReport)
    assert report.passed is True
    assert report.regressions == ()
    assert report.n_matched == 3
    assert abs(report.mean_delta) < 1e-9


def test_check_detects_regressions_on_degraded_outputs():
    snap = snapshot_baseline(GOLDEN)
    report = check_against_baseline(snap, DEGRADED, threshold=0.05)
    assert report.passed is False
    assert len(report.regressions) > 0
    assert report.mean_current_aggregate < report.mean_baseline_aggregate
    for r in report.regressions:
        assert r.delta < 0  # current < baseline
        assert -r.delta > 0.05


def test_check_threshold_zero_catches_any_drop():
    snap = snapshot_baseline(GOLDEN)
    report = check_against_baseline(snap, DEGRADED, threshold=0.0)
    assert len(report.regressions) >= len(
        check_against_baseline(snap, DEGRADED, threshold=0.10).regressions
    )


def test_check_rejects_negative_threshold():
    snap = snapshot_baseline(GOLDEN)
    with pytest.raises(ValueError):
        check_against_baseline(snap, GOLDEN, threshold=-0.1)


def test_check_reports_unmatched_inputs():
    snap = snapshot_baseline(GOLDEN)
    # Drop the first item, add a brand new one.
    drift = GOLDEN[1:] + [{"input": "Brand new question", "output": "Brand new answer"}]
    report = check_against_baseline(snap, drift)
    assert len(report.unmatched_baseline) == 1  # GOLDEN[0] missing from current
    assert len(report.unmatched_current) == 1  # the new item not in baseline
    assert report.passed is False  # unmatched baseline counts as a failure


def test_check_reorders_safely():
    snap = snapshot_baseline(GOLDEN)
    reordered = [GOLDEN[2], GOLDEN[0], GOLDEN[1]]
    report = check_against_baseline(snap, reordered)
    assert report.passed is True
    assert report.n_matched == 3


# ── BaselineStore + FileBaselineStore ───────────────────────────────────


def test_in_memory_store_save_load_list():
    store = BaselineStore()
    snap = snapshot_baseline(GOLDEN, label="initial")
    sid = store.save(snap)
    assert sid in store.list()
    assert store.load(sid).label == "initial"
    assert len(store) == 1


def test_in_memory_store_raises_keyerror_on_missing():
    store = BaselineStore()
    with pytest.raises(KeyError):
        store.load("does-not-exist")


def test_file_store_persists_across_instances(tmp_path: Path):
    snap = snapshot_baseline(GOLDEN, label="persist")
    store1 = FileBaselineStore(str(tmp_path))
    sid = store1.save(snap)
    # New instance loads from disk.
    store2 = FileBaselineStore(str(tmp_path))
    loaded = store2.load(sid)
    assert loaded.snapshot_id == sid
    assert loaded.label == "persist"
    assert loaded.n_items == 3
    # And the JSON file on disk parses as the same payload.
    raw = json.loads((tmp_path / f"{sid}.json").read_text(encoding="utf-8"))
    assert raw["snapshot_id"] == sid


def test_file_store_ignores_corrupt_files(tmp_path: Path):
    (tmp_path / "garbage.json").write_text("not json at all", encoding="utf-8")
    store = FileBaselineStore(str(tmp_path))
    assert len(store) == 0
