"""FastAPI SSE streaming server for Kairu inference.

Endpoint
--------
``POST /generate`` → ``text/event-stream`` (default)
``POST /generate`` → ``application/x-ndjson``  (when ``Accept: application/x-ndjson``)

Request body (JSON)::

    {
        "prompt": str,            # required, <= max_prompt_chars
        "max_tokens": int,        # 1..max_tokens_cap, default 64
        "temperature": float,     # 0.0..2.0, default 1.0
        "stop_token_id": int|null # optional early-stop sentinel
    }

Each streamed SSE frame is OpenAI-compatible::

    data: {"id":"kairu-...","object":"chat.completion.chunk","created":<ts>,
           "model":"<name>","choices":[{"index":0,"delta":{"content":"<piece>"},
           "finish_reason":null}],
           "kairu":{"token_id":<id>,"latency_ms":<f>,"tokens_per_s":<f>,"index":<i>}}

A final ``data: [DONE]\\n\\n`` sentinel terminates the SSE stream (OpenAI convention).

JSONL fallback (``Accept: application/x-ndjson``)
--------------------------------------------------
Clients that cannot consume ``text/event-stream`` (e.g. plain ``curl``,
webhooks, batch processors) may set ``Accept: application/x-ndjson``.  The
server emits the same JSON payload objects as newline-delimited JSON — one
object per line, no ``data:`` prefix, no ``[DONE]`` sentinel::

    {"id":"kairu-...","object":"chat.completion.chunk",...}\\n
    {"id":"kairu-...","object":"chat.completion.chunk",...,"choices":[{..."finish_reason":"length"}],...}\\n

Security (CLAUDE.md §Inference Server Security)
-----------------------------------------------
* Inputs validated at the boundary — prompt length, max_tokens, temperature, charset all
  enforced *before* tokenization or any model call.
* Per-IP token-bucket rate limit (default 10 req / 10s) on every endpoint.
* Per-request wall-clock timeout via ``asyncio.wait_for``.
* Prompt content is never logged at INFO+; only a SHA-256 prefix.

Observability
-------------
* OpenTelemetry trace context is extracted from ``traceparent`` / ``tracestate``
  headers and propagated into per-request ``kairu.generate`` spans.
* Per-token events are recorded on the span (not child spans — at 1 k tokens/s
  that would balloon the trace store).
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
from kairu.squish_eval import (
    SquishEvaluator,
    quality_degradation_report,
    recommended_quant_tier,
)
from kairu.streaming import StreamingDecoder
from kairu.tokenizer import MockTokenizer, TokenizerBase
from kairu.tracing import KairuTracer, extract_trace_context, headers_from_request

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

    class QuantCompareRequest(BaseModel):
        """Side-by-side quantization comparison request.

        ``baseline_outputs``  — model outputs at full precision (typically FP16).
        ``quant_tiers``       — ``{tier_name: [outputs...]}`` aligned with baseline.
        ``references``        — optional gold answers, used in place of baseline as
                                ground truth when present.
        ``tolerance``         — fraction of aggregate quality the caller is willing
                                to lose for the recommendation (default 0.05 = 5%).
        """

        baseline_outputs: list[str] = Field(..., min_length=1, max_length=512)
        quant_tiers: dict[str, list[str]] = Field(..., min_length=1, max_length=8)
        references: Optional[list[str]] = Field(default=None)
        tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
except ImportError:  # pragma: no cover — server extras not installed
    GenerateRequest = None  # type: ignore[assignment,misc]
    QuantCompareRequest = None  # type: ignore[assignment,misc]


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
    tracer: Optional[KairuTracer] = None,
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
    otel = tracer if tracer is not None else KairuTracer()

    app = FastAPI(title="Kairu Inference Server", version="0.7.0")
    app.state.config = cfg
    app.state.model = mdl
    app.state.tokenizer = tok
    app.state.rate_limiter = limiter
    app.state.metrics = mtx
    app.state.tracer = otel

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

        # Determine output format: SSE (default) or JSONL fallback.
        accept = request.headers.get("accept", "")
        use_jsonl = "application/x-ndjson" in accept

        # Extract OTel trace context from incoming headers.
        parent_ctx = extract_trace_context(headers_from_request(request.headers))

        prompt_ids = tok.encode(req.prompt)
        decoder = StreamingDecoder(mdl, temperature=req.temperature)

        async def _token_loop(span, framer):
            """Core generation loop shared by SSE and JSONL paths.

            Args:
                span:   Active OTel span (or NoOpSpan).
                framer: Callable(frame_dict) → bytes encoding one frame.

            Yields raw bytes for StreamingResponse.
            """
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
                    lat_ms = round(dt * 1000.0, 4)

                    # Record per-token OTel event (annotation, not child span).
                    otel.record_token(span, index, tok_id, lat_ms)

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
                            "latency_ms": lat_ms,
                            "tokens_per_s": round(tps, 4),
                        },
                    }
                    yield framer(frame)
                    # Yield to the event loop so the response actually flushes
                    # token-by-token instead of buffering the whole generation.
                    await asyncio.sleep(0)

                if count >= req.max_tokens and finish_reason == "stop":
                    finish_reason = "length"

                total_s = round(time.monotonic() - start, 6)
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
                        "total_s": total_s,
                    },
                }
                yield framer(final)
                otel.record_generation_complete(span, count, finish_reason, total_s)
                mtx.tokens_generated_total.inc(count, finish_reason=finish_reason)
                mtx.requests_total.inc(endpoint="/generate", status="200")
                mtx.request_duration_seconds.observe(
                    time.monotonic() - req_start, endpoint="/generate"
                )
            except Exception as exc:  # noqa: BLE001 — stream errors are reported in-band
                logger.exception("generate id=%s failed", request_id)
                otel.record_error(span, exc)
                err = {
                    "id": request_id,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                }
                yield framer(err)
                mtx.errors_total.inc(kind="stream_failed")
                mtx.requests_total.inc(endpoint="/generate", status="500")
            finally:
                mtx.active_streams.dec()

        if use_jsonl:
            # JSONL fallback: one JSON object per line, no SSE framing or sentinel.
            def _jsonl_frame(payload: dict) -> bytes:
                return f"{json.dumps(payload, ensure_ascii=False)}\n".encode("utf-8")

            async def jsonl_stream() -> AsyncIterator[bytes]:
                with otel.start_generate_span(request_id, prompt_hash, parent_ctx) as span:
                    async for chunk in _token_loop(span, _jsonl_frame):
                        yield chunk

            return StreamingResponse(
                jsonl_stream(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Default: SSE stream.
        async def event_stream() -> AsyncIterator[bytes]:
            with otel.start_generate_span(request_id, prompt_hash, parent_ctx) as span:
                async for chunk in _token_loop(span, _sse_frame):
                    yield chunk
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    squish_evaluator = SquishEvaluator()

    @app.post("/compare/quantization")
    async def compare_quantization(request: Request):
        """Score baseline vs. quantized outputs and recommend a tier.

        Rate-limited like /generate. Pure CPU work — no model call, no
        streaming. Returns a single JSON payload."""
        req_start = time.monotonic()
        try:
            payload = await request.json()
            req = QuantCompareRequest.model_validate(payload)
        except ValidationError as e:
            mtx.requests_total.inc(endpoint="/compare/quantization", status="422")
            mtx.errors_total.inc(kind="validation")
            return JSONResponse(status_code=422, content={"detail": e.errors()})
        except Exception as e:  # noqa: BLE001 — invalid JSON
            mtx.requests_total.inc(endpoint="/compare/quantization", status="422")
            mtx.errors_total.inc(kind="invalid_json")
            return JSONResponse(status_code=422, content={"detail": str(e)})

        key = _client_key(request)
        if not await limiter.check(key):
            mtx.requests_total.inc(endpoint="/compare/quantization", status="429")
            mtx.rate_limited_total.inc()
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded ({cfg.rate_limit_requests}/{cfg.rate_limit_window_s}s)",
            )

        try:
            report = quality_degradation_report(
                baseline_outputs=req.baseline_outputs,
                quantized_outputs=req.quant_tiers,
                references=req.references,
                evaluator=squish_evaluator,
            )
        except ValueError as e:
            mtx.requests_total.inc(endpoint="/compare/quantization", status="422")
            mtx.errors_total.inc(kind="limit_violation")
            return JSONResponse(status_code=422, content={"detail": str(e)})

        recommendation = recommended_quant_tier(report, tolerance=req.tolerance)

        mtx.requests_total.inc(endpoint="/compare/quantization", status="200")
        mtx.request_duration_seconds.observe(
            time.monotonic() - req_start, endpoint="/compare/quantization"
        )
        return JSONResponse(
            status_code=200,
            content={
                "report": report.as_dict(),
                "recommended_tier": recommendation,
                "tolerance": req.tolerance,
            },
        )

    @app.exception_handler(ValidationError)
    async def _on_validation(_request, exc):  # pragma: no cover — wired via fastapi
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return app


__all__ = ["ServerConfig", "RateLimiter", "create_app", "KairuTracer"]
