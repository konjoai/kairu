"""OpenTelemetry trace export for Kairu inference.

Instruments ``/generate`` with per-token spans and propagates W3C
``traceparent`` / ``tracestate`` context from the incoming request so
traces stitch end-to-end in Jaeger, Tempo, or any OTLP-compatible backend.

Design constraints
------------------
* Zero hard dependency on ``opentelemetry-sdk``.  The module is always
  importable; OTel is lazy-imported at tracer construction time.  If the
  package is absent the ``NoOpTracer`` stub is used transparently so the
  server still works — tracing is additive.
* No monkey-patching of the global OTel SDK registry from inside the
  module.  Callers own the SDK initialisation (``TracerProvider``,
  ``BatchSpanProcessor``, OTLP exporter) so kairu stays embeddable.

Usage (with opentelemetry-sdk installed)::

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
    )
    trace.set_tracer_provider(provider)

    from kairu.tracing import KairuTracer
    tracer = KairuTracer()               # wraps the global provider

Usage (no opentelemetry-sdk — NoOp)::

    from kairu.tracing import KairuTracer
    tracer = KairuTracer()               # falls back to NoOpTracer automatically

Wire into server::

    from kairu.tracing import KairuTracer, extract_trace_context
    tracer = KairuTracer()

    # inside a request handler:
    ctx = extract_trace_context(request.headers)
    with tracer.start_generate_span(request_id, prompt_hash, ctx) as span:
        for i, tok_id in enumerate(token_stream):
            tracer.record_token(span, i, tok_id, latency_ms)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterator, Optional

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# W3C TraceContext propagation
# ---------------------------------------------------------------------------

def extract_trace_context(headers: Dict[str, str]) -> Optional[Any]:
    """Extract an OpenTelemetry ``Context`` from incoming HTTP headers.

    Understands ``traceparent`` / ``tracestate`` (W3C TraceContext).  Returns
    ``None`` when OTel is unavailable, keeping the caller unconditional.
    """
    traceparent = headers.get("traceparent") or headers.get("Traceparent")
    if not traceparent:
        return None
    try:
        from opentelemetry.propagators.textmap import DefaultTextMapPropagator  # type: ignore
        ctx = DefaultTextMapPropagator().extract(carrier=dict(headers))
        return ctx
    except Exception:  # noqa: BLE001 — OTel absent or extraction failure
        return None


# ---------------------------------------------------------------------------
# NoOp stubs (used when opentelemetry-sdk is not installed)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Span that silently drops every operation."""

    def set_attribute(self, _key: str, _value: Any) -> "_NoOpSpan":
        return self

    def add_event(self, _name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def set_status(self, _status: Any) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *_args: Any) -> None:
        pass


class _NoOpTracer:
    """Tracer that produces only _NoOpSpan instances."""

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        **kwargs: Any,
    ) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# KairuTracer — thin facade over the OTel API (or NoOp when absent)
# ---------------------------------------------------------------------------

INSTRUMENTATION_NAME = "kairu.server"
INSTRUMENTATION_VERSION = "0.7.0"


class KairuTracer:
    """Thin OTel tracer facade with automatic NoOp fallback.

    Constructs an OTel ``Tracer`` from the globally registered
    ``TracerProvider`` if ``opentelemetry-api`` is importable; otherwise
    falls back to :class:`_NoOpTracer` so callers need zero ``try/except``
    blocks.

    Thread-safe: the underlying OTel tracer is thread-safe by spec; the
    NoOp is trivially so.
    """

    def __init__(self) -> None:
        self._tracer: Any = self._build_tracer()
        self._otel_available = not isinstance(self._tracer, _NoOpTracer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def start_generate_span(
        self,
        request_id: str,
        prompt_hash: str,
        parent_context: Optional[Any] = None,
    ) -> Generator[Any, None, None]:
        """Start a ``kairu.generate`` root span for one /generate request.

        Args:
            request_id:     Unique request ID (e.g. ``kairu-<hex16>``).
            prompt_hash:    SHA-256 prefix of the prompt (privacy-safe).
            parent_context: OTel context extracted from the incoming request
                            headers via :func:`extract_trace_context`.  When
                            provided the span is a *child* of the caller's
                            trace; when ``None`` it is a new root span.

        Yields:
            The active span (either a real OTel span or :class:`_NoOpSpan`).
        """
        kwargs: Dict[str, Any] = {}
        if parent_context is not None:
            kwargs["context"] = parent_context

        with self._tracer.start_as_current_span(
            "kairu.generate",
            **kwargs,
        ) as span:
            span.set_attribute("kairu.request_id", request_id)
            span.set_attribute("kairu.prompt_sha256_prefix", prompt_hash)
            span.set_attribute("kairu.start_time_unix_ns", time.time_ns())
            yield span

    def record_token(
        self,
        span: Any,
        index: int,
        token_id: int,
        latency_ms: float,
    ) -> None:
        """Add a per-token event to *span*.

        Events are cheap OTel annotations — they carry structured key/value
        attributes but do not create child spans (which would balloon the
        trace when generating thousands of tokens).
        """
        span.add_event(
            "kairu.token",
            attributes={
                "token.index": index,
                "token.id": token_id,
                "token.latency_ms": round(latency_ms, 4),
            },
        )

    def record_generation_complete(
        self,
        span: Any,
        tokens_generated: int,
        finish_reason: str,
        total_s: float,
    ) -> None:
        """Annotate the span with final generation statistics."""
        span.set_attribute("kairu.tokens_generated", tokens_generated)
        span.set_attribute("kairu.finish_reason", finish_reason)
        span.set_attribute("kairu.total_s", round(total_s, 6))

    def record_error(self, span: Any, exc: Exception) -> None:
        """Mark *span* as failed and attach the exception."""
        span.record_exception(exc)
        if self._otel_available:
            try:
                from opentelemetry.trace import StatusCode  # type: ignore
                span.set_status(StatusCode.ERROR, str(exc))
            except Exception:  # noqa: BLE001
                pass

    @property
    def is_noop(self) -> bool:
        """True when OpenTelemetry is unavailable and the NoOp tracer is active."""
        return not self._otel_available

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_tracer(self) -> Any:
        try:
            from opentelemetry import trace  # type: ignore
            return trace.get_tracer(
                INSTRUMENTATION_NAME,
                INSTRUMENTATION_VERSION,
            )
        except ImportError:
            return _NoOpTracer()


# ---------------------------------------------------------------------------
# Context propagation helper for use in server.py
# ---------------------------------------------------------------------------

def headers_from_request(request_headers) -> Dict[str, str]:
    """Normalise ASGI/Starlette MutableHeaders → plain dict for propagation.

    Accepts anything with ``.items()`` and returns a ``{str: str}`` dict
    safe to pass to :func:`extract_trace_context`.
    """
    return {k.lower(): v for k, v in request_headers.items()}


__all__ = [
    "KairuTracer",
    "extract_trace_context",
    "headers_from_request",
]
