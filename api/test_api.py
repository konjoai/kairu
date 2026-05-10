"""HTTP boundary tests for api.main — uses httpx ASGI transport (no live port)."""
from __future__ import annotations

import csv
import io

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import MAX_BATCH_ITEMS, MAX_TEXT_CHARS, app


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


# --------------------------------- /health ----------------------------------


async def test_health_ok(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "kairu-eval"
    assert "version" in body


# --------------------------------- /rubrics ---------------------------------


async def test_rubrics_lists_default(client: AsyncClient) -> None:
    r = await client.get("/rubrics")
    assert r.status_code == 200
    body = r.json()
    names = {rb["name"] for rb in body["rubrics"]}
    assert "default" in names
    assert "helpfulness" in names
    crit_names = {c["name"] for c in body["criteria"]}
    assert "relevance" in crit_names
    assert "safety" in crit_names


# -------------------------------- /evaluate ---------------------------------


async def test_evaluate_returns_aggregate_and_per_criterion(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate",
        json={
            "prompt": "What is photosynthesis?",
            "response": "Photosynthesis converts sunlight into glucose using chlorophyll in plant leaves.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["aggregate"] <= 1.0
    assert "scores" in body
    assert "details" in body
    assert "weights" in body
    assert body["rubric"] == "default"


async def test_evaluate_with_explicit_criteria(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate",
        json={"prompt": "p", "response": "r words here", "criteria": ["relevance", "fluency"]},
    )
    assert r.status_code == 200
    assert set(r.json()["scores"].keys()) == {"relevance", "fluency"}


async def test_evaluate_unknown_rubric_returns_422(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate", json={"prompt": "p", "response": "r", "rubric": "ghost"}
    )
    assert r.status_code == 422


async def test_evaluate_unknown_criterion_returns_422(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate", json={"prompt": "p", "response": "r", "criteria": ["nonsense"]}
    )
    assert r.status_code == 422


async def test_evaluate_missing_prompt_returns_422(client: AsyncClient) -> None:
    r = await client.post("/evaluate", json={"response": "r"})
    assert r.status_code == 422


async def test_evaluate_oversize_prompt_returns_413(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate", json={"prompt": "x" * (MAX_TEXT_CHARS + 1), "response": "r"}
    )
    assert r.status_code == 413


# --------------------------------- /compare ---------------------------------


async def test_compare_picks_winner(client: AsyncClient) -> None:
    r = await client.post(
        "/compare",
        json={
            "prompt": "Explain photosynthesis in detail",
            "response_a": "Photosynthesis is the process by which plants convert sunlight into glucose using chlorophyll in their leaves.",
            "response_b": "idk",
            "label_a": "model-a",
            "label_b": "model-b",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["winner"] == "a"
    assert body["margin"] > 0.0
    assert body["label_a"] == "model-a"
    assert body["label_b"] == "model-b"
    assert isinstance(body["per_criterion"], list)
    assert all("delta" in c for c in body["per_criterion"])


async def test_compare_tie_when_responses_equal(client: AsyncClient) -> None:
    r = await client.post(
        "/compare",
        json={"prompt": "hi", "response_a": "hello world", "response_b": "hello world"},
    )
    body = r.json()
    assert body["winner"] == "tie"
    assert body["margin"] < 0.01


async def test_compare_missing_b_returns_422(client: AsyncClient) -> None:
    r = await client.post("/compare", json={"prompt": "p", "response_a": "a"})
    assert r.status_code == 422


# ---------------------------------- /batch ----------------------------------


async def test_batch_json_default(client: AsyncClient) -> None:
    r = await client.post(
        "/batch",
        json={
            "items": [
                {"id": "a", "prompt": "q1", "response": "answer one"},
                {"id": "b", "prompt": "q2", "response": "answer two with detail"},
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert {row["id"] for row in body["results"]} == {"a", "b"}


async def test_batch_csv_format(client: AsyncClient) -> None:
    r = await client.post(
        "/batch",
        json={
            "items": [{"id": "x", "prompt": "q", "response": "an answer"}],
            "format": "csv",
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][0] == "id"
    assert rows[1][0] == "x"


async def test_batch_empty_returns_422(client: AsyncClient) -> None:
    r = await client.post("/batch", json={"items": []})
    assert r.status_code == 422


async def test_batch_oversize_returns_413(client: AsyncClient) -> None:
    items = [{"prompt": "p", "response": "r"} for _ in range(MAX_BATCH_ITEMS + 1)]
    r = await client.post("/batch", json={"items": items})
    assert r.status_code == 413


async def test_rubric_detail_returns_full_definition(client: AsyncClient) -> None:
    r = await client.get("/rubrics/helpfulness")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "helpfulness"
    assert "weights" in body
    assert "criteria" in body
    assert body["color"].startswith("#")


async def test_rubric_detail_unknown_returns_404(client: AsyncClient) -> None:
    r = await client.get("/rubrics/does_not_exist")
    assert r.status_code == 404


async def test_evaluate_named_rubric_uses_path_param(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate/rubric/safety",
        json={"prompt": "Generate test data", "response": "Email: a@b.test, no PII included."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rubric"] == "safety"
    assert "color" in body
    assert "scores" in body


async def test_evaluate_named_rubric_unknown_returns_404(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate/rubric/ghost", json={"prompt": "p", "response": "r"}
    )
    assert r.status_code == 404


async def test_evaluate_named_rubric_oversize_returns_413(client: AsyncClient) -> None:
    from api.main import MAX_TEXT_CHARS
    r = await client.post(
        "/evaluate/rubric/helpfulness",
        json={"prompt": "x" * (MAX_TEXT_CHARS + 1), "response": "r"},
    )
    assert r.status_code == 413


async def test_rubrics_list_includes_color_for_named_rubrics(client: AsyncClient) -> None:
    r = await client.get("/rubrics")
    assert r.status_code == 200
    body = r.json()
    by_name = {rb["name"]: rb for rb in body["rubrics"]}
    for name in ("helpfulness", "accuracy", "safety", "coherence",
                 "conciseness", "creativity", "groundedness", "tone"):
        assert name in by_name
        assert by_name[name]["color"].startswith("#")


async def test_batch_propagates_unknown_rubric(client: AsyncClient) -> None:
    r = await client.post(
        "/batch",
        json={
            "items": [{"prompt": "p", "response": "r"}],
            "rubric": "nonexistent",
        },
    )
    assert r.status_code == 422
