"""Saved evaluation templates — eliminate copy-paste of repeat eval configs.

A template freezes everything an `/evaluate` call needs except the actual
``(prompt, response)`` pair: rubric, criteria list, weights, and (optionally)
a judge ensemble. Apply with ``POST /evaluate?template=name`` instead of
re-sending eight fields per request.

Storage
-------
SQLite. One row per template name. Body is JSON for forward-compat with
new optional fields. Updates replace the row (last-write-wins) so the
API can use the same endpoint for create *and* update.

Why SQLite
----------
The audit log already uses SQLite and the API process is single-writer
in the common deployment. One more table costs nothing and gives us
durable, queryable, transactional persistence with zero new dependencies.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kairu.ensemble import JudgeConfig


# ─────────────────────────────────────────────────────────────────────────
# Schema + helpers
# ─────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS templates (
    name         TEXT PRIMARY KEY,
    description  TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL,
    created_utc  REAL NOT NULL,
    updated_utc  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_updated ON templates(updated_utc);
"""


@dataclass(frozen=True)
class EvaluationTemplate:
    """A frozen evaluation configuration."""

    name: str
    description: str
    rubric: Optional[str]
    criteria: Optional[List[str]]
    weights: Optional[Dict[str, float]]
    # If present, the template applies via ensemble_evaluate.
    judges: Optional[List[Dict[str, Any]]]
    created_utc: float
    updated_utc: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "rubric": self.rubric,
            "criteria": list(self.criteria) if self.criteria is not None else None,
            "weights": dict(self.weights) if self.weights is not None else None,
            "judges": [dict(j) for j in self.judges] if self.judges else None,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
        }

    def judge_configs(self) -> Optional[List[JudgeConfig]]:
        """Materialise stored judge dicts into JudgeConfig objects."""
        if not self.judges:
            return None
        out: List[JudgeConfig] = []
        for j in self.judges:
            out.append(JudgeConfig(
                name=str(j["name"]),
                rubric=j.get("rubric", "default"),
                criteria=tuple(j["criteria"]) if j.get("criteria") else None,
                weights=j.get("weights"),
                seed=j.get("seed"),
                noise=float(j.get("noise", 0.0)),
            ))
        return out


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("template name must be a non-empty string")
    if len(name) > 64:
        raise ValueError("template name must be ≤ 64 chars")
    # Conservative charset — letters, digits, dash, underscore, dot.
    for ch in name:
        if not (ch.isalnum() or ch in "-_."):
            raise ValueError(f"template name contains invalid character: {ch!r}")


# ─────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────


class TemplateStore:
    """Thread-safe SQLite-backed template store.

    Pass ``:memory:`` for ephemeral storage (tests) or a filesystem path
    for durable storage. Multiple processes pointing at the same file
    work fine for low write volumes — SQLite's default isolation is
    sufficient.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> str:
        return self._path

    def save(
        self,
        name: str,
        *,
        description: str = "",
        rubric: Optional[str] = None,
        criteria: Optional[Sequence[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
        judges: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> EvaluationTemplate:
        _validate_name(name)
        if rubric is None and not criteria and not judges:
            raise ValueError(
                "template must specify at least one of: rubric, criteria, judges"
            )
        body = json.dumps({
            "rubric": rubric,
            "criteria": list(criteria) if criteria else None,
            "weights": dict(weights) if weights else None,
            "judges": [dict(j) for j in judges] if judges else None,
        }, sort_keys=True, ensure_ascii=False)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT created_utc FROM templates WHERE name = ?", (name,),
            )
            row = cur.fetchone()
            created = row["created_utc"] if row else now
            self._conn.execute(
                "INSERT OR REPLACE INTO templates "
                "(name, description, body, created_utc, updated_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, description, body, created, now),
            )
        return self.get(name)

    def get(self, name: str) -> EvaluationTemplate:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name, description, body, created_utc, updated_utc "
                "FROM templates WHERE name = ?",
                (name,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(name)
        body = json.loads(row["body"])
        return EvaluationTemplate(
            name=row["name"],
            description=row["description"] or "",
            rubric=body.get("rubric"),
            criteria=body.get("criteria"),
            weights=body.get("weights"),
            judges=body.get("judges"),
            created_utc=float(row["created_utc"]),
            updated_utc=float(row["updated_utc"]),
        )

    def delete(self, name: str) -> bool:
        """Returns True iff the template existed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM templates WHERE name = ?", (name,)
            )
            return cur.rowcount > 0

    def list(self) -> List[EvaluationTemplate]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name, description, body, created_utc, updated_utc "
                "FROM templates ORDER BY updated_utc DESC"
            )
            rows = cur.fetchall()
        out: List[EvaluationTemplate] = []
        for row in rows:
            body = json.loads(row["body"])
            out.append(EvaluationTemplate(
                name=row["name"],
                description=row["description"] or "",
                rubric=body.get("rubric"),
                criteria=body.get("criteria"),
                weights=body.get("weights"),
                judges=body.get("judges"),
                created_utc=float(row["created_utc"]),
                updated_utc=float(row["updated_utc"]),
            ))
        return out

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM templates")
            return int(cur.fetchone()["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def open_default_template_store() -> TemplateStore:
    """Resolve a store from ``KAIRU_TEMPLATE_DB``; in-memory if unset."""
    path = os.environ.get("KAIRU_TEMPLATE_DB", ":memory:")
    return TemplateStore(path)


__all__ = [
    "EvaluationTemplate",
    "TemplateStore",
    "open_default_template_store",
]
