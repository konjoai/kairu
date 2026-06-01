"""Prompt library — save and recall named prompts across evaluations.

Eval workflows reuse the same prompts constantly: a smoke-test set used
in CI, a golden corpus for regression testing, a list of safety probes
for adversarial checks. Re-pasting them into every request is friction
and a source of drift. This module owns the table.

Schema
------
``prompts(name PK, text, description, tags_json, created_utc, updated_utc)``

INSERT OR REPLACE semantics — last-write-wins on ``name``, with
``created_utc`` preserved across updates. Tags are an arbitrary string
list, indexed only at query time (typical libraries are <1000 entries).
``KAIRU_PROMPT_DB`` env switches in-memory → file.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    name         TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    tags_json    TEXT NOT NULL DEFAULT '[]',
    created_utc  REAL NOT NULL,
    updated_utc  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompts_updated ON prompts(updated_utc);
"""


@dataclass(frozen=True)
class Prompt:
    """One saved prompt."""

    name: str
    text: str
    description: str
    tags: List[str]
    created_utc: float
    updated_utc: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "description": self.description,
            "tags": list(self.tags),
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
        }


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("prompt name must be a non-empty string")
    if len(name) > 64:
        raise ValueError("prompt name must be ≤ 64 chars")
    for ch in name:
        if not (ch.isalnum() or ch in "-_."):
            raise ValueError(f"prompt name contains invalid character: {ch!r}")


def _normalise_tags(tags: Optional[Sequence[str]]) -> List[str]:
    if not tags:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for t in tags:
        if not isinstance(t, str):
            raise TypeError("each tag must be a string")
        t = t.strip().lower()
        if not t or t in seen:
            continue
        if len(t) > 32:
            raise ValueError(f"tag '{t[:20]}…' exceeds 32 chars")
        seen.add(t)
        out.append(t)
    return out


class PromptStore:
    """Thread-safe SQLite-backed prompt store."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None,
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
        text: str,
        description: str = "",
        tags: Optional[Sequence[str]] = None,
    ) -> Prompt:
        _validate_name(name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > 200_000:
            raise ValueError("text must be ≤ 200000 chars")
        tag_list = _normalise_tags(tags)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT created_utc FROM prompts WHERE name = ?", (name,),
            )
            row = cur.fetchone()
            created = row["created_utc"] if row else now
            self._conn.execute(
                "INSERT OR REPLACE INTO prompts "
                "(name, text, description, tags_json, created_utc, updated_utc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, text, description, json.dumps(tag_list), created, now),
            )
        return self.get(name)

    def get(self, name: str) -> Prompt:
        with self._lock:
            row = self._conn.execute(
                "SELECT name, text, description, tags_json, created_utc, updated_utc "
                "FROM prompts WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise KeyError(name)
        return self._row_to_prompt(row)

    def delete(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM prompts WHERE name = ?", (name,))
            return cur.rowcount > 0

    def list(self, *, tag: Optional[str] = None) -> List[Prompt]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, text, description, tags_json, created_utc, updated_utc "
                "FROM prompts ORDER BY updated_utc DESC"
            ).fetchall()
        out = [self._row_to_prompt(r) for r in rows]
        if tag is not None:
            t = tag.strip().lower()
            out = [p for p in out if t in p.tags]
        return out

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM prompts"
            ).fetchone()["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_prompt(row: sqlite3.Row) -> Prompt:
        try:
            tags = list(json.loads(row["tags_json"] or "[]"))
        except (TypeError, ValueError):
            tags = []
        return Prompt(
            name=row["name"],
            text=row["text"],
            description=row["description"] or "",
            tags=tags,
            created_utc=float(row["created_utc"]),
            updated_utc=float(row["updated_utc"]),
        )


def open_default_prompt_store() -> PromptStore:
    """Resolve a store from ``KAIRU_PROMPT_DB``; in-memory if unset."""
    path = os.environ.get("KAIRU_PROMPT_DB", ":memory:")
    return PromptStore(path)


__all__ = [
    "Prompt",
    "PromptStore",
    "open_default_prompt_store",
]
