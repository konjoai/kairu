"""Rubric Marketplace — community rubric sharing with domain tagging and signatures.

A marketplace entry is a ``(name, version, domain, description, rubric, signature,
source_url)`` tuple backed by SQLite. Entries are identified by ``(name, version)``
and signed with a deterministic SHA-256 so consumers can verify provenance.

``seed_community_rubrics`` populates a fresh store with four built-in community
rubrics (medical, legal, creative_writing, code_review) that use only the criteria
defined in ``kairu.evaluation.CRITERIA``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "DOMAINS",
    "MarketplaceEntry",
    "MarketplaceStore",
    "open_default_marketplace_store",
    "seed_community_rubrics",
    "compute_signature",
]

logger = logging.getLogger("kairu.marketplace")

DOMAINS: tuple[str, ...] = (
    "medical",
    "legal",
    "creative_writing",
    "code_review",
    "general",
    "safety",
    "education",
)

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS marketplace (
        name        TEXT    NOT NULL,
        version     TEXT    NOT NULL,
        domain      TEXT    NOT NULL,
        description TEXT    NOT NULL,
        rubric_json TEXT    NOT NULL,
        signature   TEXT    NOT NULL,
        source_url  TEXT    NOT NULL DEFAULT '',
        created_utc REAL    NOT NULL,
        PRIMARY KEY (name, version)
    )
"""


@dataclass(frozen=True)
class MarketplaceEntry:
    """Immutable marketplace rubric record."""

    name: str
    version: str
    domain: str
    description: str
    rubric: dict  # criterion → weight mapping
    signature: str  # SHA-256(name:version:sorted-rubric-json)
    source_url: str
    created_utc: float


def compute_signature(name: str, version: str, rubric: dict) -> str:
    """Return a deterministic SHA-256 hex digest for a rubric payload."""
    payload = f"{name}:{version}:{json.dumps(rubric, sort_keys=True)}"
    return hashlib.sha256(payload.encode()).hexdigest()


class MarketplaceStore:
    """SQLite-backed community rubric store.

    Thread-safe for concurrent reads (WAL mode). Writes are serialised by the
    global Python GIL; for high-write deployments wrap in a connection pool.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_CREATE_TABLE)
        self._db.commit()

    # ── public interface ──────────────────────────────────────────────────

    def publish(
        self,
        name: str,
        version: str,
        domain: str,
        description: str,
        rubric: dict,
        source_url: str = "",
    ) -> MarketplaceEntry:
        """Insert or replace a marketplace entry. Raises ``ValueError`` on bad input."""
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        if not rubric:
            raise ValueError("rubric must contain at least one criterion")
        sig = compute_signature(name, version, rubric)
        rubric_json = json.dumps(rubric, sort_keys=True)
        now = time.time()
        self._db.execute(
            "INSERT OR REPLACE INTO marketplace VALUES (?,?,?,?,?,?,?,?)",
            (name, version, domain, description, rubric_json, sig, source_url, now),
        )
        self._db.commit()
        logger.debug(
            "published marketplace rubric %s@%s domain=%s", name, version, domain
        )
        return MarketplaceEntry(
            name=name,
            version=version,
            domain=domain,
            description=description,
            rubric=rubric,
            signature=sig,
            source_url=source_url,
            created_utc=now,
        )

    def list_entries(
        self,
        domain: Optional[str] = None,
        q: Optional[str] = None,
    ) -> List[MarketplaceEntry]:
        """Return entries newest-first, optionally filtered by domain and keyword."""
        sql = "SELECT * FROM marketplace WHERE 1=1"
        params: list = []
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        if q:
            sql += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{q}%", f"%{q}%"])
        sql += " ORDER BY created_utc DESC"
        rows = self._db.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get(
        self, name: str, version: Optional[str] = None
    ) -> Optional[MarketplaceEntry]:
        """Return the named entry (latest version if *version* is omitted)."""
        if version:
            row = self._db.execute(
                "SELECT * FROM marketplace WHERE name=? AND version=?",
                (name, version),
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM marketplace WHERE name=? ORDER BY created_utc DESC LIMIT 1",
                (name,),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()

    # ── private helpers ───────────────────────────────────────────────────

    def _row_to_entry(self, row: tuple) -> MarketplaceEntry:
        return MarketplaceEntry(
            name=row[0],
            version=row[1],
            domain=row[2],
            description=row[3],
            rubric=json.loads(row[4]),
            signature=row[5],
            source_url=row[6],
            created_utc=row[7],
        )


def open_default_marketplace_store() -> MarketplaceStore:
    """Open the marketplace store at ``KAIRU_MARKETPLACE_DB`` (defaults to ``:memory:``)."""
    db_path = os.environ.get("KAIRU_MARKETPLACE_DB", ":memory:")
    return MarketplaceStore(db_path=db_path)


# ── community rubrics ─────────────────────────────────────────────────────────
# All criteria are from kairu.evaluation.CRITERIA so they can be imported to
# the local RUBRIC_REGISTRY without validation errors.

_COMMUNITY_RUBRICS: tuple[dict, ...] = (
    {
        "name": "medical_qa",
        "version": "1.0.0",
        "domain": "medical",
        "description": (
            "Evaluates medical Q&A for clinical accuracy, completeness, "
            "patient safety, and specificity."
        ),
        "rubric": {
            "relevance": 0.30,
            "completeness": 0.35,
            "safety": 0.25,
            "specificity": 0.10,
        },
        "source_url": "https://github.com/konjoai/kairu/wiki/rubric-medical-qa",
    },
    {
        "name": "legal_analysis",
        "version": "1.0.0",
        "domain": "legal",
        "description": (
            "Scores legal analysis on relevance, logical coherence, "
            "statutory specificity, and completeness."
        ),
        "rubric": {
            "relevance": 0.30,
            "coherence": 0.25,
            "specificity": 0.30,
            "completeness": 0.15,
        },
        "source_url": "https://github.com/konjoai/kairu/wiki/rubric-legal-analysis",
    },
    {
        "name": "creative_writing",
        "version": "1.0.0",
        "domain": "creative_writing",
        "description": (
            "Assesses creative writing quality — fluency of prose, narrative "
            "coherence, originality via specificity, and tight conciseness."
        ),
        "rubric": {
            "coherence": 0.30,
            "fluency": 0.35,
            "specificity": 0.20,
            "conciseness": 0.15,
        },
        "source_url": "https://github.com/konjoai/kairu/wiki/rubric-creative-writing",
    },
    {
        "name": "code_review",
        "version": "1.0.0",
        "domain": "code_review",
        "description": (
            "Evaluates code review responses for correctness-relevance, "
            "coherent reasoning, safety, and completeness of feedback."
        ),
        "rubric": {
            "relevance": 0.25,
            "coherence": 0.20,
            "safety": 0.30,
            "completeness": 0.25,
        },
        "source_url": "https://github.com/konjoai/kairu/wiki/rubric-code-review",
    },
)


def seed_community_rubrics(store: MarketplaceStore) -> None:
    """Seed *store* with built-in community rubrics (idempotent — skips existing)."""
    for spec in _COMMUNITY_RUBRICS:
        if store.get(spec["name"], version=spec["version"]) is None:
            store.publish(
                name=spec["name"],
                version=spec["version"],
                domain=spec["domain"],
                description=spec["description"],
                rubric=spec["rubric"],
                source_url=spec["source_url"],
            )
