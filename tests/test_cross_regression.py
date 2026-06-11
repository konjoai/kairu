"""Tests for kairu.cross_regression — compare_models and /regression endpoint."""

from __future__ import annotations

import pytest

from kairu.leaderboard import LeaderboardStore
from kairu.cross_regression import (
    CrossModelReport,
    compare_models,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from api.main import create_app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_store(model_a_score: float, model_b_score: float) -> LeaderboardStore:
    """Seed a fresh in-memory leaderboard with two models."""
    store = LeaderboardStore()
    store.record(
        model="model-a", prompt="p1", rubric_name="default", aggregate=model_a_score
    )
    store.record(
        model="model-b", prompt="p1", rubric_name="default", aggregate=model_b_score
    )
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_compare_models_returns_cross_model_report() -> None:
    """compare_models returns a CrossModelReport instance."""
    store = _make_store(0.8, 0.7)
    report = compare_models("model-a", "model-b", store)
    assert isinstance(report, CrossModelReport)


def test_aggregate_delta_is_score_b_minus_score_a() -> None:
    """aggregate_delta equals score_b - score_a."""
    store = _make_store(0.8, 0.5)
    report = compare_models("model-a", "model-b", store)
    assert report.aggregate_delta == pytest.approx(0.5 - 0.8, abs=1e-9)


def test_has_regressions_true_when_delta_below_negative_threshold() -> None:
    """has_regressions is True when score_b < score_a - threshold."""
    store = _make_store(0.9, 0.5)
    report = compare_models("model-a", "model-b", store, threshold=0.02)
    assert report.has_regressions is True
    assert len(report.regressions) == 1
    assert report.regressions[0].criterion == "aggregate"


def test_has_regressions_false_when_delta_within_threshold() -> None:
    """has_regressions is False when |delta| <= threshold."""
    store = _make_store(0.8, 0.81)
    report = compare_models("model-a", "model-b", store, threshold=0.02)
    assert report.has_regressions is False
    assert len(report.neutral) == 1


def test_improvements_filled_when_score_b_above_threshold() -> None:
    """improvements is non-empty when score_b > score_a + threshold."""
    store = _make_store(0.5, 0.9)
    report = compare_models("model-a", "model-b", store, threshold=0.02)
    assert len(report.improvements) == 1
    assert report.improvements[0].delta > 0


def test_neutral_when_abs_delta_within_threshold() -> None:
    """neutral is non-empty when |delta| <= threshold."""
    store = _make_store(0.8, 0.81)
    report = compare_models("model-a", "model-b", store, threshold=0.02)
    assert len(report.neutral) == 1
    assert len(report.regressions) == 0
    assert len(report.improvements) == 0


def test_raises_value_error_when_model_a_not_found() -> None:
    """compare_models raises ValueError when model_a is not in the leaderboard."""
    store = _make_store(0.8, 0.7)
    with pytest.raises(ValueError, match="model-x"):
        compare_models("model-x", "model-b", store)


def test_raises_value_error_when_model_b_not_found() -> None:
    """compare_models raises ValueError when model_b is not in the leaderboard."""
    store = _make_store(0.8, 0.7)
    with pytest.raises(ValueError, match="model-y"):
        compare_models("model-a", "model-y", store)


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def regression_app():
    """Create a fresh app with a seeded leaderboard."""
    store = LeaderboardStore()
    store.record(model="alpha", prompt="hello", rubric_name="default", aggregate=0.9)
    store.record(model="beta", prompt="hello", rubric_name="default", aggregate=0.6)
    return create_app(leaderboard_store=store)


@pytest.fixture
async def client(regression_app) -> AsyncClient:
    """Async HTTP client bound to the seeded regression app."""
    async with AsyncClient(
        transport=ASGITransport(app=regression_app), base_url="http://t"
    ) as c:
        yield c


async def test_api_regression_returns_200_and_correct_structure(
    client: AsyncClient,
) -> None:
    """GET /regression with two known models returns 200 and correct structure."""
    r = await client.get("/regression?model_a=alpha&model_b=beta")
    assert r.status_code == 200
    body = r.json()
    assert body["model_a"] == "alpha"
    assert body["model_b"] == "beta"
    assert "aggregate_delta" in body
    assert "has_regressions" in body
    assert "n_compared" in body
    assert isinstance(body["regressions"], list)
    assert isinstance(body["improvements"], list)
    assert isinstance(body["neutral"], list)
    assert body["aggregate_delta"] == pytest.approx(0.6 - 0.9, abs=1e-6)
    assert body["has_regressions"] is True


async def test_api_regression_unknown_model_returns_404(
    client: AsyncClient,
) -> None:
    """GET /regression with an unknown model returns 404."""
    r = await client.get("/regression?model_a=alpha&model_b=ghost-model")
    assert r.status_code == 404
