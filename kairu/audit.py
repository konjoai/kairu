"""Immutable append-only audit log for evaluations.

Stores every ``/evaluate``, ``/evaluate/rubric/{name}`` and ``/compare``
call to a SQLite database. The schema is intentionally narrow and the
public API exposes only ``record``, ``query``, ``count``, ``export_csv``
— no UPDATE, no DELETE methods. The underlying SQLite file is created
with ``WAL`` journaling so concurrent readers (the dashboard) cannot
block writers (the API).

Append-only is **enforced at two layers**:

1. The Python class exposes no mutation methods beyond ``record``.
2. The schema installs triggers that ``RAISE(ABORT, ...)`` on any
   ``UPDATE`` or ``DELETE`` against the ``evaluations`` table — so even
   an operator with a raw ``sqlite3`` shell cannot rewrite history
   without dropping the trigger themselves (an audit-detectable event).

Rows
----
- ``id``               auto-increment primary key
- ``timestamp_utc``    ISO 8601 with microseconds, UTC
- ``input_hash``       SHA-256 of ``prompt||0x1f||response[||0x1f||response_b]``
- ``rubric_name``      e.g. ``"helpfulness"``
- ``rubric_version``   SemVer; see :mod:`kairu.evaluation`
- ``judge_model``      free text — ``"kairu-heuristic"`` for the in-tree
                       scorer, an OpenAI model id for hosted judges, etc.
- ``endpoint``         ``"/evaluate"`` / ``"/compare"`` / etc.
- ``scores_json``      JSON-encoded ``{criterion: score, ...}``
- ``reasoning_json``   JSON-encoded reasoning trace (may be empty ``"{}"``)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc   TEXT    NOT NULL,
    input_hash      TEXT    NOT NULL,
    rubric_name     TEXT    NOT NULL,
    rubric_version  TEXT    NOT NULL,
    judge_model     TEXT    NOT NULL,
    endpoint        TEXT    NOT NULL,
    scores_json     TEXT    NOT NULL,
    reasoning_json  TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_evaluations_rubric    ON evaluations(rubric_name, rubric_version);
CREATE INDEX IF NOT EXISTS idx_evaluations_hash      ON evaluations(input_hash);

-- Append-only triggers. Any UPDATE or DELETE against `evaluations` aborts
-- the transaction with a clear error. The triggers themselves can be
-- dropped (we cannot prevent the SQLite admin from doing that with raw
-- access) but doing so is a detectable, auditable schema change.
CREATE TRIGGER IF NOT EXISTS evaluations_no_update
BEFORE UPDATE ON evaluations
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS evaluations_no_delete
BEFORE DELETE ON evaluations
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: DELETE forbidden');
END;
"""


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def hash_inputs(prompt: str, response: str, response_b: Optional[str] = None) -> str:
    """SHA-256 over the input fields with a unit separator (0x1f) between
    them. Stable across processes and trivial to recompute."""
    sep = b"\x1f"
    parts = [prompt.encode("utf-8"), response.encode("utf-8")]
    if response_b is not None:
        parts.append(response_b.encode("utf-8"))
    return hashlib.sha256(sep.join(parts)).hexdigest()


@dataclass(frozen=True)
class AuditRecord:
    id: int
    timestamp_utc: str
    input_hash: str
    rubric_name: str
    rubric_version: str
    judge_model: str
    endpoint: str
    scores: Dict[str, float]
    reasoning: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp_utc,
            "input_hash": self.input_hash,
            "rubric_name": self.rubric_name,
            "rubric_version": self.rubric_version,
            "judge_model": self.judge_model,
            "endpoint": self.endpoint,
            "scores": self.scores,
            "reasoning": self.reasoning,
        }


class AppendOnlyError(RuntimeError):
    """Raised when an UPDATE or DELETE is attempted against the audit table."""


class AuditLog:
    """Thin synchronous wrapper around the audit-log SQLite database.

    Thread-safe: writes are serialised by an internal lock and the
    underlying connection uses ``check_same_thread=False``. The API runs
    in a single FastAPI worker by default; if you scale to N workers,
    point them all at the same DB file — SQLite WAL handles concurrent
    writers correctly for low/medium volumes (our target).
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,           # autocommit; we manage txns explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ── public surface ────────────────────────────────────────────────

    @property
    def path(self) -> str:
        return self._path

    def record(
        self,
        *,
        input_hash: str,
        rubric_name: str,
        rubric_version: str,
        judge_model: str,
        endpoint: str,
        scores: Mapping[str, float],
        reasoning: Optional[Mapping[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> int:
        """Append one row. Returns the new row id."""
        ts = timestamp_utc or _now_iso_utc()
        payload = (
            ts,
            input_hash,
            rubric_name,
            rubric_version,
            judge_model,
            endpoint,
            json.dumps(dict(scores), sort_keys=True, ensure_ascii=False),
            json.dumps(dict(reasoning or {}), sort_keys=True, ensure_ascii=False),
        )
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO evaluations
                   (timestamp_utc, input_hash, rubric_name, rubric_version,
                    judge_model, endpoint, scores_json, reasoning_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )
            return int(cur.lastrowid)

    def query(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        rubric_name: Optional[str] = None,
        rubric_version: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """Paginated time-range query, newest first."""
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be in [1, 10000]")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        clauses: List[str] = []
        args: List[Any] = []
        if start is not None:
            clauses.append("timestamp_utc >= ?")
            args.append(start)
        if end is not None:
            clauses.append("timestamp_utc <= ?")
            args.append(end)
        if rubric_name is not None:
            clauses.append("rubric_name = ?")
            args.append(rubric_name)
        if rubric_version is not None:
            clauses.append("rubric_version = ?")
            args.append(rubric_version)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT * FROM evaluations {where} "
            f"ORDER BY timestamp_utc DESC, id DESC LIMIT ? OFFSET ?"
        )
        args.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        rubric_name: Optional[str] = None,
        rubric_version: Optional[str] = None,
    ) -> int:
        clauses: List[str] = []
        args: List[Any] = []
        if start is not None:
            clauses.append("timestamp_utc >= ?")
            args.append(start)
        if end is not None:
            clauses.append("timestamp_utc <= ?")
            args.append(end)
        if rubric_name is not None:
            clauses.append("rubric_name = ?")
            args.append(rubric_name)
        if rubric_version is not None:
            clauses.append("rubric_version = ?")
            args.append(rubric_version)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT COUNT(*) AS n FROM evaluations {where}"
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return int(row["n"])

    def export_csv(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        rubric_name: Optional[str] = None,
        rubric_version: Optional[str] = None,
    ) -> str:
        """Emit the matching rows as CSV. ``scores`` and ``reasoning`` are
        serialised as JSON strings inside their columns — the CSV stays
        flat, no nested arrays."""
        rows = self.query(
            start=start, end=end,
            rubric_name=rubric_name, rubric_version=rubric_version,
            limit=10_000, offset=0,
        )
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "id", "timestamp_utc", "input_hash", "rubric_name",
            "rubric_version", "judge_model", "endpoint",
            "scores_json", "reasoning_json",
        ])
        for r in rows:
            writer.writerow([
                r.id, r.timestamp_utc, r.input_hash, r.rubric_name,
                r.rubric_version, r.judge_model, r.endpoint,
                json.dumps(r.scores, sort_keys=True, ensure_ascii=False),
                json.dumps(r.reasoning, sort_keys=True, ensure_ascii=False),
            ])
        return buf.getvalue()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> AuditRecord:
        return AuditRecord(
            id=int(r["id"]),
            timestamp_utc=r["timestamp_utc"],
            input_hash=r["input_hash"],
            rubric_name=r["rubric_name"],
            rubric_version=r["rubric_version"],
            judge_model=r["judge_model"],
            endpoint=r["endpoint"],
            scores=json.loads(r["scores_json"]),
            reasoning=json.loads(r["reasoning_json"]),
        )


def open_default_audit() -> AuditLog:
    """Open the audit log at the path in ``KAIRU_AUDIT_DB`` (default ``:memory:``).

    The HTTP layer calls this at app startup. Production deployments set
    ``KAIRU_AUDIT_DB=/var/lib/kairu/audit.db`` (a writable volume).
    """
    path = os.environ.get("KAIRU_AUDIT_DB", ":memory:")
    return AuditLog(path)


__all__ = [
    "AuditLog",
    "AuditRecord",
    "AppendOnlyError",
    "hash_inputs",
    "open_default_audit",
]
