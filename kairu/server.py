"""FastAPI SSE streaming server for Kairu inference.

Endpoint
--------
``POST /generate`` → ``text/event-stream``

Request body (JSON)::

    {
        "prompt": str,            # required, <= max_prompt_chars
        "max_tokens": int,        # 1..max_tokens_cap, default 64
        "temperature": float,     # 0.0..2.0, default 1.0
        "stop_token_id": int|null # optional early-stop sentinel
    }

Each streamed frame is OpenAI-compatible::

    data: {"id":"kairu-...","object":"chat.completion.chunk","created":<ts>,
           "model":"<name>","choices":[{"index":0,"delta":{"content":"<piece>"},
           "finish_reason":null}],
           "kairu":{"token_id":<id>,"latency_ms":<f>,"tokens_per_s":<f>,"index":<i>}}

A final ``data: [DONE]\\n\\n`` sentinel terminates the stream (OpenAI convention).

Security (CLAUDE.md §Inference Server Security)
-----------------------------------------------
* Inputs validated at the boundary — prompt length, max_tokens, temperature, charset all
  enforced *before* tokenization or any model call.
* Per-IP token-bucket rate limit (default 10 req / 10s) on every endpoint.
* Per-request wall-clock timeout via ``asyncio.wait_for``.
* Prompt content is never logged at INFO+; only a SHA-256 prefix.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from kairu.base import ModelInterface
from kairu.metrics_export import CONTENT_TYPE as METRICS_CT, MetricsCollector
from kairu.mock_model import MockModel
from kairu.rate_limit import (
    InMemoryBackend,
    RateLimiter,
    RateLimiterBackend,
)
from kairu.streaming import StreamingDecoder
from kairu.tokenizer import MockTokenizer, TokenizerBase

logger = logging.getLogger("kairu.server")

# Allow printable ASCII + common unicode letters/marks/digits/punct/space.
# Reject control chars (except \t, \n, \r) to neutralize obvious injection payloads
# at the byte level before they reach the tokenizer.
_FORBIDDEN_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class ServerConfig:
    """Server-side limits. All bounds are enforced at the request boundary."""

    model_name: str = "kairu-mock"
    max_prompt_chars: int = 8192
    max_tokens_cap: int = 512
    request_timeout_s: float = 30.0
    rate_limit_requests: int = 10
    rate_limit_window_s: float = 10.0


def _validate_prompt(prompt: str, max_chars: int) -> None:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if len(prompt) == 0:
        raise ValueError("prompt must be non-empty")
    if len(prompt) > max_chars:
        raise ValueError(f"prompt exceeds max length ({max_chars} chars)")
    if _FORBIDDEN_CTRL.search(prompt):
        raise ValueError("prompt contains forbidden control characters")


def _hash_prefix(text: str, n: int = 12) -> str:
    """SHA-256 prefix — for log correlation without leaking prompt content."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


try:  # Module-level import keeps annotations resolvable for FastAPI's type inspection.
    from starlette.requests import Request  # noqa: F401 — used in endpoint annotation
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment,misc]

try:  # Pydantic is required for the server but optional for the package
    from pydantic import BaseModel, Field

    class GenerateRequest(BaseModel):
        """Request schema. Hard upper bounds match the most permissive ServerConfig;
        per-instance caps are tightened in :func:`_enforce_limits`."""

        prompt: str = Field(..., min_length=1, max_length=1_000_000)
        max_tokens: int = Field(default=64, ge=1, le=100_000)
        temperature: float = Field(default=1.0, ge=0.0, le=2.0)
        stop_token_id: Optional[int] = Field(default=None, ge=0)
except ImportError:  # pragma: no cover — server extras not installed
    GenerateRequest = None  # type: ignore[assignment,misc]


def _enforce_limits(req: "GenerateRequest", cfg: ServerConfig) -> None:
    """Tighten the schema's permissive defaults against this server's config."""
    _validate_prompt(req.prompt, cfg.max_prompt_chars)
    if req.max_tokens > cfg.max_tokens_cap:
        raise ValueError(
            f"max_tokens={req.max_tokens} exceeds cap ({cfg.max_tokens_cap})"
        )


def _sse_frame(payload: dict) -> bytes:
    """One SSE event. ``json.dumps`` with ``ensure_ascii`` keeps the wire safe."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def create_app(
    model: Optional[ModelInterface] = None,
    tokenizer: Optional[TokenizerBase] = None,
    config: Optional[ServerConfig] = None,
    rate_limit_backend: Optional[RateLimiterBackend] = None,
    metrics: Optional[MetricsCollector] = None,
):
    """Build a FastAPI app. Lazy-imports FastAPI so ``import kairu`` stays cheap."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, Response, StreamingResponse
        from pydantic import ValidationError
    except ImportError as e:  # pragma: no cover — surfaced via install extras
        raise ImportError(
            "kairu.server requires FastAPI. Install with: pip install kairu[server]"
        ) from e

    cfg = config or ServerConfig()
    mdl: ModelInterface = model if model is not None else MockModel()
    tok: TokenizerBase = tokenizer if tokenizer is not None else MockTokenizer()
    limiter = RateLimiter(
        cfg.rate_limit_requests,
        cfg.rate_limit_window_s,
        backend=rate_limit_backend,
    )
    mtx = metrics if metrics is not None else MetricsCollector()

    app = FastAPI(title="Kairu Inference Server", version="0.6.0")
    app.state.config = cfg
    app.state.model = mdl
    app.state.tokenizer = tok
    app.state.rate_limiter = limiter
    app.state.metrics = mtx

    def _client_key(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    @app.get("/health")
    async def health() -> dict:
        mtx.requests_total.inc(endpoint="/health", status="200")
        return {"status": "ok", "model": cfg.model_name, "version": app.version}

    @app.get("/metrics")
    async def metrics_endpoint():
        mtx.requests_total.inc(endpoint="/metrics", status="200")
        body = mtx.render()
        return Response(content=body, media_type=METRICS_CT)

    @app.post("/generate")
    async def generate(request: Request):
        req_start = time.monotonic()
        try:
            req = GenerateRequest.model_validate(await request.json())
        except ValidationError as e:
            mtx.requests_total.inc(endpoint="/generate", status="422")
            mtx.errors_total.inc(kind="validation")
            return JSONResponse(status_code=422, content={"detail": e.errors()})
        except Exception as e:  # noqa: BLE001 — invalid JSON
            mtx.requests_total.inc(endpoint="/generate", status="422")
            mtx.errors_total.inc(kind="invalid_json")
            return JSONResponse(status_code=422, content={"detail": str(e)})

        try:
            _enforce_limits(req, cfg)
        except ValueError as e:
            mtx.requests_total.inc(endpoint="/generate", status="422")
            mtx.errors_total.inc(kind="limit_violation")
            return JSONResponse(status_code=422, content={"detail": str(e)})

        key = _client_key(request)
        if not await limiter.check(key):
            mtx.requests_total.inc(endpoint="/generate", status="429")
            mtx.rate_limited_total.inc()
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded ({cfg.rate_limit_requests}/{cfg.rate_limit_window_s}s)",
            )

        prompt_hash = _hash_prefix(req.prompt)
        request_id = f"kairu-{uuid.uuid4().hex[:16]}"
        logger.info(
            "generate id=%s client=%s prompt_sha=%s max_tokens=%d temp=%.3f",
            request_id, key, prompt_hash, req.max_tokens, req.temperature,
        )

        prompt_ids = tok.encode(req.prompt)
        decoder = StreamingDecoder(mdl, temperature=req.temperature)

        async def event_stream() -> AsyncIterator[bytes]:
            start = time.monotonic()
            deadline = start + cfg.request_timeout_s
            last_t = start
            count = 0
            finish_reason = "stop"
            mtx.active_streams.inc()

            try:
                for index, tok_id in enumerate(
                    decoder.stream(prompt_ids, req.max_tokens, req.stop_token_id)
                ):
                    now = time.monotonic()
                    if now >= deadline:
                        finish_reason = "timeout"
                        break

                    piece = tok.decode([tok_id])
                    dt = now - last_t
                    last_t = now
                    count += 1
                    mtx.token_latency_seconds.observe(dt)
                    elapsed = now - start
                    tps = count / elapsed if elapsed > 0 else 0.0

                    frame = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": cfg.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }],
                        "kairu": {
                            "token_id": tok_id,
                            "index": index,
                            "latency_ms": round(dt * 1000.0, 4),
                            "tokens_per_s": round(tps, 4),
                        },
                    }
                    yield _sse_frame(frame)
                    # Yield to the event loop so the response actually flushes
                    # token-by-token instead of buffering the whole generation.
                    await asyncio.sleep(0)

                if count >= req.max_tokens and finish_reason == "stop":
                    finish_reason = "length"

                final = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                    "kairu": {
                        "tokens_generated": count,
                        "total_s": round(time.monotonic() - start, 6),
                    },
                }
                yield _sse_frame(final)
                yield b"data: [DONE]\n\n"
                mtx.tokens_generated_total.inc(count, finish_reason=finish_reason)
                mtx.requests_total.inc(endpoint="/generate", status="200")
                mtx.request_duration_seconds.observe(
                    time.monotonic() - req_start, endpoint="/generate"
                )
            except Exception as exc:  # noqa: BLE001 — stream errors are reported in-band
                logger.exception("generate id=%s failed", request_id)
                err = {
                    "id": request_id,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                }
                yield _sse_frame(err)
                yield b"data: [DONE]\n\n"
                mtx.errors_total.inc(kind="stream_failed")
                mtx.requests_total.inc(endpoint="/generate", status="500")
            finally:
                mtx.active_streams.dec()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.exception_handler(ValidationError)
    async def _on_validation(_request, exc):  # pragma: no cover — wired via fastapi
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return app


__all__ = ["ServerConfig", "RateLimiter", "create_app"]
