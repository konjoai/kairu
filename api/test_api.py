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


# ─────────────────────────── /benchmarks ────────────────────────────────


async def test_benchmarks_index_lists_criteria(client: AsyncClient) -> None:
    r = await client.get("/benchmarks")
    assert r.status_code == 200
    body = r.json()
    assert "criteria" in body
    assert "corpus_size" in body
    assert body["corpus_size"] == 1000
    assert "relevance" in body["criteria"]


async def test_benchmarks_detail_returns_quantiles(client: AsyncClient) -> None:
    r = await client.get("/benchmarks/relevance")
    assert r.status_code == 200
    body = r.json()
    for k in ("p25", "p50", "p75", "p90", "p99", "mean", "stdev", "histogram", "n"):
        assert k in body
    assert body["n"] == 1000
    assert body["p25"] <= body["p50"] <= body["p75"] <= body["p90"] <= body["p99"]


async def test_benchmarks_detail_404_for_unknown(client: AsyncClient) -> None:
    r = await client.get("/benchmarks/totally-not-a-thing")
    assert r.status_code == 404


# ─────────────────────────── /evaluate gains benchmarks + audit ─────────


async def test_evaluate_returns_rubric_version_and_benchmarks(client: AsyncClient) -> None:
    r = await client.post(
        "/evaluate",
        json={"prompt": "What is the GIL?",
              "response": "It serializes Python bytecode execution.",
              "rubric": "accuracy"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "rubric_version" in body
    assert "benchmarks" in body
    assert "audit_id" in body
    assert isinstance(body["audit_id"], int)
    for crit, block in body["benchmarks"].items():
        assert "you" in block
        assert "rank" in block
        assert "p50" in block


# ─────────────────────────── /compare significance ──────────────────────


async def test_compare_returns_significance_block(client: AsyncClient) -> None:
    r = await client.post(
        "/compare",
        json={
            "prompt": "Explain the GIL briefly.",
            "response_a": "CPython serializes bytecode; threads do not parallelize CPU work, "
                          "but they overlap blocking I/O. Use multiprocessing for CPU parallelism.",
            "response_b": "Maybe.",
            "rubric": "helpfulness",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "significance" in body
    assert "statistical_winner" in body
    sig = body["significance"]
    for k in ("n", "mean_diff", "p_value", "effect_size", "effect_label",
              "confidence_interval", "winner"):
        assert k in sig
    assert sig["mean_diff"] > 0


async def test_compare_records_audit_row(client: AsyncClient) -> None:
    before = await client.get("/audit")
    before_total = before.json()["total"]
    await client.post(
        "/compare",
        json={
            "prompt": "Compare these.",
            "response_a": "The first reply contains specific details.",
            "response_b": "Generic answer.",
        },
    )
    after = await client.get("/audit")
    after_total = after.json()["total"]
    assert after_total == before_total + 1


# ─────────────────────────── /audit ────────────────────────────────────


async def test_audit_listing_after_evaluation(client: AsyncClient) -> None:
    await client.post("/evaluate", json={"prompt": "p", "response": "r"})
    r = await client.get("/audit")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    item = body["items"][0]
    for k in ("id", "timestamp_utc", "input_hash", "rubric_name",
              "rubric_version", "judge_model", "endpoint", "scores"):
        assert k in item


async def test_audit_filter_by_rubric_name(client: AsyncClient) -> None:
    await client.post("/evaluate",
        json={"prompt": "p", "response": "r", "rubric": "safety"})
    r = await client.get("/audit", params={"rubric_name": "safety"})
    body = r.json()
    assert all(item["rubric_name"] == "safety" for item in body["items"])


async def test_audit_csv_export(client: AsyncClient) -> None:
    await client.post("/evaluate", json={"prompt": "p", "response": "r"})
    r = await client.get("/audit.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    reader = csv.reader(io.StringIO(r.text))
    rows = list(reader)
    assert rows[0][0] == "id"
    assert len(rows) >= 2


async def test_audit_rejects_oversized_limit(client: AsyncClient) -> None:
    r = await client.get("/audit", params={"limit": 99999})
    assert r.status_code == 422


# ─────────────────────────── /rubrics version registry ─────────────────


async def test_rubrics_list_includes_version(client: AsyncClient) -> None:
    r = await client.get("/rubrics")
    body = r.json()
    for rb in body["rubrics"]:
        assert "version" in rb
        assert "versions" in rb
        assert rb["version"] in rb["versions"]


async def test_register_new_rubric_version(client: AsyncClient) -> None:
    r = await client.post(
        "/rubrics",
        json={
            "name": "helpfulness",
            "description": "Tuned for backend-engineer responses.",
            "criteria": ["relevance", "completeness", "specificity"],
            "weights": {"relevance": 0.5, "completeness": 0.3, "specificity": 0.2},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "helpfulness"
    parts = body["version"].split(".")
    assert parts[0] == "1"
    assert len(parts) == 3


async def test_register_rubric_rejects_unknown_criterion(client: AsyncClient) -> None:
    r = await client.post(
        "/rubrics",
        json={
            "name": "test-rubric",
            "criteria": ["does-not-exist"],
            "weights": {"does-not-exist": 1.0},
        },
    )
    assert r.status_code == 422


async def test_register_rubric_rejects_mismatched_keys(client: AsyncClient) -> None:
    r = await client.post(
        "/rubrics",
        json={
            "name": "test-mismatch",
            "criteria": ["relevance", "fluency"],
            "weights": {"relevance": 0.5, "safety": 0.5},
        },
    )
    assert r.status_code == 422
