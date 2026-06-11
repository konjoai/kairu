"""Human feedback store for kairu evaluation results.

Stores per-criterion thumbs-up/down overrides on evaluation rows.
SQLite-backed; ``KAIRU_FEEDBACK_DB`` env switches ``:memory:`` → file.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass

__all__ = [
    "HumanFeedback",
    "FeedbackStore",
    "open_default_feedback_store",
]


@dataclass(frozen=True)
class HumanFeedback:
    """A single human override on one criterion of an evaluation row."""

    eval_id: int
    criterion: str
    vote: int  # +1 (agree) or -1 (disagree)
    note: str
    timestamp_utc: float


class FeedbackStore:
    """Thread-safe SQLite store for human feedback records."""

    _CREATE = """
    CREATE TABLE IF NOT EXISTS feedback (
        eval_id       INTEGER NOT NULL,
        criterion     TEXT    NOT NULL,
        vote          INTEGER NOT NULL CHECK(vote IN (1, -1)),
        note          TEXT    NOT NULL DEFAULT '',
        timestamp_utc REAL    NOT NULL,
        PRIMARY KEY (eval_id, criterion)
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_eval ON feedback(eval_id);
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialise the store at ``db_path`` (default: in-memory)."""
        self._db_path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.executescript(self._CREATE)
        self._con.commit()

    def record(
        self,
        eval_id: int,
        criterion: str,
        vote: int,
        note: str = "",
    ) -> HumanFeedback:
        """Record or update a feedback vote for one criterion.

        ``vote`` must be +1 (agree/thumbs-up) or -1 (disagree/thumbs-down).
        Raises ``ValueError`` for invalid vote values.
        """
        if vote not in (1, -1):
            raise ValueError(f"vote must be +1 or -1, got {vote!r}")
        ts = time.time()
        self._con.execute(
            """INSERT INTO feedback (eval_id, criterion, vote, note, timestamp_utc)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(eval_id, criterion) DO UPDATE SET
                 vote=excluded.vote, note=excluded.note,
                 timestamp_utc=excluded.timestamp_utc""",
            (eval_id, criterion, vote, note, ts),
        )
        self._con.commit()
        return HumanFeedback(eval_id, criterion, vote, note, ts)

    def get(self, eval_id: int) -> list[HumanFeedback]:
        """Return all feedback records for the given ``eval_id``."""
        rows = self._con.execute(
            "SELECT eval_id, criterion, vote, note, timestamp_utc FROM feedback WHERE eval_id=?",
            (eval_id,),
        ).fetchall()
        return [HumanFeedback(*r) for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._con.close()


def open_default_feedback_store() -> FeedbackStore:
    """Open a ``FeedbackStore`` resolving path from ``KAIRU_FEEDBACK_DB`` env.

    Defaults to ``:memory:`` when the env var is unset.
    """
    return FeedbackStore(os.environ.get("KAIRU_FEEDBACK_DB", ":memory:"))
