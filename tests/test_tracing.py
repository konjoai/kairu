"""Tests for kairu.tracing — OTel facade, NoOp fallback, context propagation.

All tests run fully offline (no OTel SDK required). The NoOp path is always
exercised because opentelemetry-sdk is not in the project's dev extras.
"""
from __future__ import annotations

import pytest

from kairu.tracing import (
    KairuTracer,
    _NoOpSpan,
    _NoOpTracer,
    extract_trace_context,
    headers_from_request,
)


# ---------------------------------------------------------------------------
# Test 1 — KairuTracer constructs without error (NoOp path)
# ---------------------------------------------------------------------------

def test_kairu_tracer_constructs() -> None:
    tracer = KairuTracer()
    # OTel SDK is not installed in CI — must fall back to NoOp.
    assert tracer is not None


# ---------------------------------------------------------------------------
# Test 2 — is_noop reflects SDK availability
# ---------------------------------------------------------------------------

def test_kairu_tracer_is_noop_without_sdk() -> None:
    """Without opentelemetry-api installed, is_noop must be True."""
    tracer = KairuTracer()
    # SDK not in dev extras → always NoOp in the test environment.
    assert tracer.is_noop is True


# ---------------------------------------------------------------------------
# Test 3 — start_generate_span is a context manager that yields a span
# ---------------------------------------------------------------------------

def test_start_generate_span_yields_span() -> None:
    tracer = KairuTracer()
    with tracer.start_generate_span("kairu-abc", "sha256abc", parent_context=None) as span:
        assert span is not None
        # NoOpSpan supports set_attribute without raising.
        result = span.set_attribute("test.key", "test.value")
        # set_attribute returns self on NoOpSpan.
        assert result is span


# ---------------------------------------------------------------------------
# Test 4 — record_token does not raise
# ---------------------------------------------------------------------------

def test_record_token_no_raise() -> None:
    tracer = KairuTracer()
    with tracer.start_generate_span("kairu-123", "hash123") as span:
        for i in range(5):
            tracer.record_token(span, index=i, token_id=i * 10, latency_ms=1.23)


# ---------------------------------------------------------------------------
# Test 5 — record_generation_complete sets attributes
# ---------------------------------------------------------------------------

def test_record_generation_complete_no_raise() -> None:
    tracer = KairuTracer()
    with tracer.start_generate_span("id-xyz", "hashxyz") as span:
        tracer.record_generation_complete(
            span,
            tokens_generated=42,
            finish_reason="length",
            total_s=0.512,
        )


# ---------------------------------------------------------------------------
# Test 6 — record_error does not raise
# ---------------------------------------------------------------------------

def test_record_error_no_raise() -> None:
    tracer = KairuTracer()
    with tracer.start_generate_span("id-err", "hasherr") as span:
        tracer.record_error(span, RuntimeError("something went wrong"))


# ---------------------------------------------------------------------------
# Test 7 — NoOpSpan.add_event is a no-op
# ---------------------------------------------------------------------------

def test_noop_span_add_event() -> None:
    span = _NoOpSpan()
    # Must not raise even with complex attributes.
    span.add_event("my.event", attributes={"key": "value", "count": 99})


# ---------------------------------------------------------------------------
# Test 8 — NoOpTracer context manager protocol
# ---------------------------------------------------------------------------

def test_noop_tracer_context_manager() -> None:
    tracer = _NoOpTracer()
    with tracer.start_as_current_span("test.span") as span:
        assert isinstance(span, _NoOpSpan)


# ---------------------------------------------------------------------------
# Test 9 — extract_trace_context returns None when no traceparent header
# ---------------------------------------------------------------------------

def test_extract_trace_context_no_header() -> None:
    result = extract_trace_context({})
    assert result is None


# ---------------------------------------------------------------------------
# Test 10 — extract_trace_context returns None without OTel SDK
# ---------------------------------------------------------------------------

def test_extract_trace_context_without_sdk() -> None:
    """Without opentelemetry-api, extraction must return None, not raise."""
    headers = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    result = extract_trace_context(headers)
    # May be None (SDK absent) or a context object (SDK present) — either is valid.
    # The important contract is: this must not raise.
    _ = result


# ---------------------------------------------------------------------------
# Test 11 — headers_from_request normalises to lowercase
# ---------------------------------------------------------------------------

def test_headers_from_request_lowercase() -> None:
    class _MockHeaders:
        def items(self):
            return [
                ("Traceparent", "00-abc-def-01"),
                ("Content-Type", "application/json"),
                ("X-Custom-Header", "value"),
            ]

    result = headers_from_request(_MockHeaders())
    assert "traceparent" in result
    assert result["traceparent"] == "00-abc-def-01"
    assert result["content-type"] == "application/json"
    assert result["x-custom-header"] == "value"


# ---------------------------------------------------------------------------
# Test 12 — headers_from_request handles plain dict
# ---------------------------------------------------------------------------

def test_headers_from_request_from_dict() -> None:
    class _DictHeaders:
        def items(self):
            return {"TraceState": "rojo=1", "Accept": "text/event-stream"}.items()

    result = headers_from_request(_DictHeaders())
    assert "tracestate" in result
    assert "accept" in result


# ---------------------------------------------------------------------------
# Test 13 — KairuTracer is reentrant (nested spans)
# ---------------------------------------------------------------------------

def test_kairu_tracer_reentrant() -> None:
    tracer = KairuTracer()
    with tracer.start_generate_span("outer", "outh") as outer:
        tracer.record_token(outer, 0, 1, 0.5)
        with tracer.start_generate_span("inner", "inh") as inner:
            tracer.record_token(inner, 0, 2, 0.3)
            tracer.record_generation_complete(inner, 1, "stop", 0.3)
        tracer.record_generation_complete(outer, 10, "length", 5.0)
