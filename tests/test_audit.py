"""Tests for kairu.audit — immutable SQLite audit log."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

from kairu.audit import AuditLog, AuditRecord, hash_inputs, open_default_audit


@pytest.fixture
def log(tmp_path):
    db = tmp_path / "audit.db"
    log = AuditLog(str(db))
    yield log
    log.close()


def test_record_returns_increasing_ids(log):
    id1 = log.record(
        input_hash="h1", rubric_name="r", rubric_version="1.0.0",
        judge_model="kairu", endpoint="/evaluate", scores={"a": 0.5},
    )
    id2 = log.record(
        input_hash="h2", rubric_name="r", rubric_version="1.0.0",
        judge_model="kairu", endpoint="/evaluate", scores={"a": 0.6},
    )
    assert id2 > id1


def test_query_returns_newest_first(log):
    for i in range(5):
        log.record(
            input_hash=f"h{i}", rubric_name="r", rubric_version="1.0.0",
            judge_model="kairu", endpoint="/evaluate", scores={"x": i / 10},
        )
        time.sleep(0.001)
    rows = log.query(limit=10)
    assert len(rows) == 5
    ids = [r.id for r in rows]
    assert ids == sorted(ids, reverse=True)


def test_query_pagination(log):
    for i in range(7):
        log.record(
            input_hash=f"h{i}", rubric_name="r", rubric_version="1.0.0",
            judge_model="kairu", endpoint="/evaluate", scores={"x": i / 10},
        )
    page1 = log.query(limit=3, offset=0)
    page2 = log.query(limit=3, offset=3)
    page3 = log.query(limit=3, offset=6)
    assert len(page1) == 3
    assert len(page2) == 3
    assert len(page3) == 1
    all_ids = {r.id for r in page1 + page2 + page3}
    assert len(all_ids) == 7


def test_filter_by_rubric_name_and_version(log):
    log.record(input_hash="h1", rubric_name="alpha", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .5})
    log.record(input_hash="h2", rubric_name="beta", rubric_version="2.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .6})
    log.record(input_hash="h3", rubric_name="alpha", rubric_version="1.1.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .7})
    assert log.count(rubric_name="alpha") == 2
    assert log.count(rubric_name="alpha", rubric_version="1.0.0") == 1
    assert log.count(rubric_version="2.0.0") == 1


def test_count_matches_query_total(log):
    for i in range(10):
        log.record(input_hash=f"h{i}", rubric_name="r", rubric_version="1.0.0",
                   judge_model="k", endpoint="/evaluate", scores={"x": .1})
    assert log.count() == 10
    rows = log.query(limit=1000)
    assert len(rows) == 10


def test_scores_round_trip(log):
    payload = {"helpfulness": 0.823, "safety": 0.91, "tone": 0.65}
    log.record(input_hash="h", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores=payload,
               reasoning={"note": "test"})
    rows = log.query(limit=1)
    assert rows[0].scores == payload
    assert rows[0].reasoning == {"note": "test"}


def test_append_only_blocks_update(log):
    log.record(input_hash="h", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .5})
    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("UPDATE evaluations SET rubric_name='hacked' WHERE id=1")


def test_append_only_blocks_delete(log):
    log.record(input_hash="h", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .5})
    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("DELETE FROM evaluations WHERE id=1")


def test_invalid_limit_raises(log):
    with pytest.raises(ValueError):
        log.query(limit=0)
    with pytest.raises(ValueError):
        log.query(limit=20_000)
    with pytest.raises(ValueError):
        log.query(offset=-1)


def test_csv_export_round_trip(log):
    log.record(input_hash="h1", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .5})
    log.record(input_hash="h2", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .6})
    csv_body = log.export_csv()
    lines = csv_body.splitlines()
    # Header + 2 rows
    assert len(lines) == 3
    header = lines[0].split(",")
    for col in ("id", "timestamp_utc", "input_hash", "rubric_name",
                "rubric_version", "judge_model", "endpoint",
                "scores_json", "reasoning_json"):
        assert col in header


def test_hash_inputs_stable():
    h1 = hash_inputs("prompt", "response")
    h2 = hash_inputs("prompt", "response")
    assert h1 == h2
    assert len(h1) == 64


def test_hash_inputs_distinguishes_response_b():
    h1 = hash_inputs("p", "r")
    h2 = hash_inputs("p", "r", "rb")
    assert h1 != h2


def test_hash_inputs_changes_on_prompt():
    h1 = hash_inputs("p1", "r")
    h2 = hash_inputs("p2", "r")
    assert h1 != h2


def test_time_range_filter(log):
    log.record(input_hash="old", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .1},
               timestamp_utc="2020-01-01T00:00:00.000000Z")
    log.record(input_hash="mid", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .5},
               timestamp_utc="2024-06-15T12:00:00.000000Z")
    log.record(input_hash="new", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .9},
               timestamp_utc="2026-12-31T00:00:00.000000Z")
    in_range = log.query(start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z")
    assert {r.input_hash for r in in_range} == {"mid"}


def test_open_default_audit_uses_env(tmp_path, monkeypatch):
    db = tmp_path / "env.db"
    monkeypatch.setenv("KAIRU_AUDIT_DB", str(db))
    log = open_default_audit()
    assert log.path == str(db)
    log.close()
    assert db.exists()


def test_record_to_dict_serialisable(log):
    log.record(input_hash="h", rubric_name="r", rubric_version="1.0.0",
               judge_model="k", endpoint="/evaluate", scores={"x": .5})
    rows = log.query(limit=1)
    json.dumps(rows[0].to_dict())  # must not raise
