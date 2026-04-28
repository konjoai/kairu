"""Tests for kairu.streaming.StreamingDecoder — 8 tests. Uses MockModel only."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from kairu.mock_model import MockModel
from kairu.streaming import StreamingDecoder

PROMPT = [10, 20, 30]


def _make_decoder(temperature: float = 1.0) -> StreamingDecoder:
    return StreamingDecoder(MockModel(), temperature=temperature)


def test_stream_yields_ints():
    dec = _make_decoder()
    tokens = list(dec.stream(PROMPT, max_new_tokens=5))
    assert all(isinstance(t, int) for t in tokens)


def test_stream_count_respects_max_new_tokens():
    dec = _make_decoder()
    for n in (1, 5, 15):
        tokens = list(dec.stream(PROMPT, max_new_tokens=n))
        assert len(tokens) <= n, f"Expected <= {n}, got {len(tokens)}"


def test_stream_stop_token_stops_early():
    dec = _make_decoder(temperature=0.0)
    # Generate one token with greedy to find what gets produced first
    first_tok = next(dec.stream(PROMPT, max_new_tokens=10))
    # Re-run with that token as the stop token — must stop after the first yield
    dec2 = _make_decoder(temperature=0.0)
    tokens = list(dec2.stream(PROMPT, max_new_tokens=20, stop_token_id=first_tok))
    assert len(tokens) == 1
    assert tokens[0] == first_tok


def test_stream_is_iterator():
    dec = _make_decoder()
    result = dec.stream(PROMPT, max_new_tokens=5)
    assert isinstance(result, Iterator)


def test_generate_returns_list():
    dec = _make_decoder()
    result = dec.generate(PROMPT, max_new_tokens=5)
    assert isinstance(result, list)


def test_generate_length_bounded():
    dec = _make_decoder()
    result = dec.generate(PROMPT, max_new_tokens=7)
    assert len(result) <= 7


def test_streaming_deterministic():
    dec1 = StreamingDecoder(MockModel(), temperature=1.0)
    dec2 = StreamingDecoder(MockModel(), temperature=1.0)
    out1 = dec1.generate(PROMPT, max_new_tokens=10)
    out2 = dec2.generate(PROMPT, max_new_tokens=10)
    assert out1 == out2, "Same seed + same prompt must yield identical output"


def test_streaming_temperature_zero_is_greedy():
    """With temperature=0, _sample always picks argmax; two runs must agree."""
    dec1 = StreamingDecoder(MockModel(), temperature=0.0)
    dec2 = StreamingDecoder(MockModel(), temperature=0.0)
    out1 = dec1.generate(PROMPT, max_new_tokens=10)
    out2 = dec2.generate(PROMPT, max_new_tokens=10)
    assert out1 == out2
    # Verify that each token equals argmax of the logits
    model = MockModel()
    tokens = list(PROMPT)
    for tok in out1:
        logits = model.next_token_logits(tokens)
        assert tok == int(np.argmax(logits)), "Greedy token must equal argmax"
        tokens.append(tok)
