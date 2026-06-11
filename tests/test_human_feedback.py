"""Tests for kairu.human_feedback — FeedbackStore, HumanFeedback, and API endpoints."""

from __future__ import annotations

import pytest

from kairu.human_feedback import (
    FeedbackStore,
    HumanFeedback,
    open_default_feedback_store,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from api.main import create_app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for FeedbackStore
# ─────────────────────────────────────────────────────────────────────────────


def test_feedback_store_construction() -> None:
    """FeedbackStore can be created in-memory without error."""
    store = FeedbackStore()
    assert store is not None
    store.close()


def test_record_stores_and_returns_correct_feedback() -> None:
    """record() returns a HumanFeedback with the submitted values."""
    store = FeedbackStore()
    fb = store.record(1, "relevance", 1, "looks right")
    assert fb.eval_id == 1
    assert fb.criterion == "relevance"
    assert fb.vote == 1
    assert fb.note == "looks right"
    store.close()


def test_get_returns_empty_list_for_unknown_eval_id() -> None:
    """get() returns [] when no feedback exists for the given eval_id."""
    store = FeedbackStore()
    assert store.get(9999) == []
    store.close()


def test_multiple_criteria_for_same_eval_id() -> None:
    """Multiple criteria can be recorded for the same eval_id."""
    store = FeedbackStore()
    store.record(42, "relevance", 1)
    store.record(42, "fluency", -1)
    store.record(42, "safety", 1)
    records = store.get(42)
    criteria = {r.criterion for r in records}
    assert criteria == {"relevance", "fluency", "safety"}
    store.close()


def test_vote_update_overwrites_existing_record() -> None:
    """Recording the same eval_id + criterion twice updates the vote."""
    store = FeedbackStore()
    store.record(7, "coherence", 1, "good")
    store.record(7, "coherence", -1, "changed mind")
    records = store.get(7)
    assert len(records) == 1
    assert records[0].vote == -1
    assert records[0].note == "changed mind"
    store.close()


def test_invalid_vote_zero_raises_value_error() -> None:
    """vote=0 raises ValueError."""
    store = FeedbackStore()
    with pytest.raises(ValueError, match="vote must be"):
        store.record(1, "relevance", 0)
    store.close()


def test_invalid_vote_positive_two_raises_value_error() -> None:
    """vote=2 raises ValueError."""
    store = FeedbackStore()
    with pytest.raises(ValueError, match="vote must be"):
        store.record(1, "relevance", 2)
    store.close()


def test_invalid_vote_string_raises_value_error() -> None:
    """vote='up' raises ValueError."""
    store = FeedbackStore()
    with pytest.raises(ValueError, match="vote must be"):
        store.record(1, "relevance", "up")  # type: ignore[arg-type]
    store.close()


def test_record_preserves_note() -> None:
    """The note field is stored and returned correctly."""
    store = FeedbackStore()
    note = "This response was off-topic but grammatically correct."
    fb = store.record(3, "relevance", -1, note)
    assert fb.note == note
    retrieved = store.get(3)
    assert retrieved[0].note == note
    store.close()


def test_timestamp_utc_is_positive_float() -> None:
    """timestamp_utc is a positive float (epoch seconds)."""
    store = FeedbackStore()
    fb = store.record(5, "safety", 1)
    assert isinstance(fb.timestamp_utc, float)
    assert fb.timestamp_utc > 0
    store.close()


def test_human_feedback_is_frozen() -> None:
    """HumanFeedback dataclass is frozen — attribute assignment raises."""
    fb = HumanFeedback(
        eval_id=1, criterion="fluency", vote=1, note="", timestamp_utc=1.0
    )
    with pytest.raises((AttributeError, TypeError)):
        fb.vote = -1  # type: ignore[misc]


def test_open_default_feedback_store_returns_feedback_store() -> None:
    """open_default_feedback_store() returns a FeedbackStore instance."""
    store = open_default_feedback_store()
    assert isinstance(store, FeedbackStore)
    store.close()


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def feedback_app():
    """Create a fresh app with an isolated in-memory feedback store."""
    store = FeedbackStore()
    application = create_app(feedback_store=store)
    return application


@pytest.fixture
async def client(feedback_app) -> AsyncClient:
    """Async HTTP client bound to the feedback-enabled app."""
    async with AsyncClient(
        transport=ASGITransport(app=feedback_app), base_url="http://t"
    ) as c:
        yield c


async def test_api_post_feedback_returns_200_and_recorded_true(
    client: AsyncClient,
) -> None:
    """POST /eval/{id}/feedback returns 200 with recorded: True."""
    r = await client.post(
        "/eval/10/feedback",
        json={"criterion": "relevance", "vote": 1, "note": "great"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["recorded"] is True
    assert body["eval_id"] == 10
    assert body["criterion"] == "relevance"
    assert body["vote"] == 1


async def test_api_get_feedback_returns_200_and_correct_structure(
    client: AsyncClient,
) -> None:
    """GET /eval/{id}/feedback returns 200 with correct response structure."""
    # Seed some feedback first
    await client.post(
        "/eval/20/feedback",
        json={"criterion": "fluency", "vote": -1, "note": "awkward phrasing"},
    )
    r = await client.get("/eval/20/feedback")
    assert r.status_code == 200
    body = r.json()
    assert body["eval_id"] == 20
    assert body["count"] == 1
    assert len(body["records"]) == 1
    record = body["records"][0]
    assert record["criterion"] == "fluency"
    assert record["vote"] == -1
    assert record["note"] == "awkward phrasing"
    assert "timestamp_utc" in record
