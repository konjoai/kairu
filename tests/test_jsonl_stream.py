"""Tests for the JSONL streaming fallback in kairu.server.

When the client sends ``Accept: application/x-ndjson`` the server must
emit newline-delimited JSON instead of SSE frames. All tests run fully
offline with MockModel.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from kairu.server import ServerConfig, create_app  # noqa: E402


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _parse_ndjson(body: str) -> list[dict]:
    """Parse NDJSON body into a list of dicts, skipping blank lines."""
    result = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            result.append(json.loads(line))
    return result


# ---------------------------------------------------------------------------
# Test 1 — JSONL response has correct Content-Type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_content_type() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 3},
            headers={"accept": "application/x-ndjson"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")


# ---------------------------------------------------------------------------
# Test 2 — JSONL response has no SSE "data:" prefix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_no_sse_prefix() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 3},
            headers={"accept": "application/x-ndjson"},
        )
    for line in r.text.splitlines():
        if line.strip():
            assert not line.startswith("data:"), f"SSE prefix found in JSONL line: {line!r}"


# ---------------------------------------------------------------------------
# Test 3 — JSONL has no [DONE] sentinel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_no_done_sentinel() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 3},
            headers={"accept": "application/x-ndjson"},
        )
    assert "[DONE]" not in r.text


# ---------------------------------------------------------------------------
# Test 4 — each JSONL line is valid JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_each_line_is_valid_json() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello world", "max_tokens": 4},
            headers={"accept": "application/x-ndjson"},
        )
    frames = _parse_ndjson(r.text)
    assert len(frames) >= 2  # at least 1 token frame + 1 final frame


# ---------------------------------------------------------------------------
# Test 5 — JSONL frames carry OpenAI-compatible chunk shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_chunk_shape() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello world", "max_tokens": 3},
            headers={"accept": "application/x-ndjson"},
        )
    frames = _parse_ndjson(r.text)
    # All but the final frame should be token chunks.
    token_frames = [f for f in frames if f.get("choices", [{}])[0].get("finish_reason") is None]
    for frame in token_frames:
        assert frame["object"] == "chat.completion.chunk"
        assert "choices" in frame
        assert "kairu" in frame
        assert "token_id" in frame["kairu"]
        assert "latency_ms" in frame["kairu"]


# ---------------------------------------------------------------------------
# Test 6 — JSONL final frame has finish_reason
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_final_frame_has_finish_reason() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 3},
            headers={"accept": "application/x-ndjson"},
        )
    frames = _parse_ndjson(r.text)
    final = frames[-1]
    finish_reason = final["choices"][0]["finish_reason"]
    assert finish_reason in ("length", "stop", "timeout")


# ---------------------------------------------------------------------------
# Test 7 — SSE still works when Accept is not application/x-ndjson
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sse_default_when_accept_is_not_ndjson() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 2},
            headers={"accept": "text/event-stream"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "data:" in r.text
    assert r.text.endswith("data: [DONE]\n\n")


# ---------------------------------------------------------------------------
# Test 8 — JSONL tokens_generated matches max_tokens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_tokens_generated_matches_max_tokens() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "test", "max_tokens": 5},
            headers={"accept": "application/x-ndjson"},
        )
    frames = _parse_ndjson(r.text)
    final = frames[-1]
    assert final["kairu"]["tokens_generated"] == 5


# ---------------------------------------------------------------------------
# Test 9 — JSONL validation still rejects bad inputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_validation_rejects_empty_prompt() -> None:
    app = create_app()
    async with _client(app) as c:
        r = await c.post(
            "/generate",
            json={"prompt": "", "max_tokens": 1},
            headers={"accept": "application/x-ndjson"},
        )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Test 10 — JSONL rate limit still returns 429
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_rate_limit_still_enforced() -> None:
    cfg = ServerConfig(rate_limit_requests=1, rate_limit_window_s=60.0)
    app = create_app(config=cfg)
    async with _client(app) as c:
        r1 = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 1},
            headers={"accept": "application/x-ndjson"},
        )
        r2 = await c.post(
            "/generate",
            json={"prompt": "hello", "max_tokens": 1},
            headers={"accept": "application/x-ndjson"},
        )
    assert r1.status_code == 200
    assert r2.status_code == 429
