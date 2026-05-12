"""Streaming text generation via SSE (Server-Sent Events).

Produces OpenAI-compatible SSE chunks:
  data: {"id": "...", "choices": [{"delta": {"content": "token"}, "index": 0, "finish_reason": null}]}
  ...
  data: {"id": "...", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
  data: [DONE]

The endpoint accepts a prompt and streams synthetic tokens (MockModel) or
real tokens (HFModel if configured). Shield runs before streaming begins.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Optional

from kairu.base import ModelInterface
from kairu.streaming import StreamingDecoder
from kairu.tokenizer import MockTokenizer, TokenizerBase

logger = logging.getLogger("kairu.streaming_api")


@dataclass
class StreamingConfig:
    """Configuration for a single streaming generation request."""

    max_tokens: int = 256
    temperature: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: int = 42
    stream_chunk_delay_ms: float = 0.0  # 0 in tests, >0 for realistic UI feel


@dataclass(frozen=True)
class StreamChunk:
    """A single SSE chunk in an OpenAI-compatible stream."""

    id: str
    content: Optional[str]  # None on the final [finish] chunk
    finish_reason: Optional[str]  # None mid-stream; "stop" or "error" on final
    index: int = 0

    def to_dict(self) -> dict:
        """Return an OpenAI-compatible chunk dict."""
        delta: dict = {} if self.content is None else {"content": self.content}
        return {
            "id": self.id,
            "choices": [
                {
                    "delta": delta,
                    "index": self.index,
                    "finish_reason": self.finish_reason,
                }
            ],
        }

    def to_sse_line(self) -> str:
        """Return 'data: {json}\\n\\n' for SSE protocol."""
        return f"data: {json.dumps(self.to_dict(), ensure_ascii=False)}\n\n"


def _make_request_id() -> str:
    """Generate a short unique request ID: 'gen_<8 hex chars>'."""
    return f"gen_{uuid.uuid4().hex[:8]}"


class TokenStreamer:
    """Generates StreamChunks from a ModelInterface.

    Uses kairu's existing StreamingDecoder (kairu/streaming.py) under the hood.
    Adds stop-sequence detection and max_tokens cap.

    The generator NEVER raises — any exception yields a final error chunk
    with finish_reason='error' and content=None, then stops.
    """

    def __init__(self, model: ModelInterface, config: StreamingConfig) -> None:
        """Initialise the streamer with a model and streaming config."""
        self._model = model
        self._config = config

    def _make_decoder(self) -> StreamingDecoder:
        """Build a StreamingDecoder seeded by config.seed."""
        import numpy as np

        decoder = StreamingDecoder(self._model, temperature=self._config.temperature)
        decoder._rng = np.random.default_rng(self._config.seed)
        return decoder

    def _stop_hit(self, accumulated: str) -> bool:
        """Return True if any stop sequence appears in accumulated text."""
        return any(seq in accumulated for seq in self._config.stop_sequences)

    def stream(
        self,
        prompt: str,
        tokenizer: Optional[TokenizerBase] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Yield StreamChunks. Final chunk has finish_reason='stop'. Never raises."""
        tok = tokenizer if tokenizer is not None else MockTokenizer()
        request_id = _make_request_id()

        try:
            yield from self._generate(prompt, tok, request_id)
        except Exception:  # noqa: BLE001 — stream errors are reported in-band
            logger.warning("TokenStreamer.stream raised; emitting error chunk", exc_info=True)
            yield StreamChunk(id=request_id, content=None, finish_reason="error")

    def _generate(
        self,
        prompt: str,
        tok: TokenizerBase,
        request_id: str,
    ) -> Generator[StreamChunk, None, None]:
        """Inner generation loop — may raise; caller converts to error chunk."""
        decoder = self._make_decoder()
        prompt_ids = tok.encode(prompt)
        accumulated = ""

        for tok_id in decoder.stream(prompt_ids, self._config.max_tokens):
            piece = tok.decode([tok_id])
            accumulated += piece
            yield StreamChunk(id=request_id, content=piece, finish_reason=None)
            if self._stop_hit(accumulated):
                break

        yield StreamChunk(id=request_id, content=None, finish_reason="stop")


__all__ = ["StreamingConfig", "StreamChunk", "TokenStreamer"]
