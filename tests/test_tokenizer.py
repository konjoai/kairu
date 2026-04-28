"""Tests for kairu.tokenizer — 8 tests. No model loading required."""
from __future__ import annotations

import pytest

from kairu.tokenizer import MockTokenizer, TokenizerBase


def test_mock_encode_returns_list():
    tok = MockTokenizer()
    result = tok.encode("hello world")
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_mock_encode_empty_string():
    tok = MockTokenizer()
    result = tok.encode("")
    assert result == []


def test_mock_decode_returns_str():
    tok = MockTokenizer()
    result = tok.decode([1, 2, 3])
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_vocab_size():
    tok = MockTokenizer(vocab_size=500)
    assert tok.vocab_size() == 500


def test_mock_encode_deterministic():
    tok = MockTokenizer()
    text = "the quick brown fox"
    ids1 = tok.encode(text)
    ids2 = tok.encode(text)
    assert ids1 == ids2


def test_mock_encode_different_texts_differ():
    tok = MockTokenizer()
    ids_a = tok.encode("hello world")
    ids_b = tok.encode("foo bar baz")
    assert ids_a != ids_b


def test_tokenizer_base_is_abstract():
    with pytest.raises(TypeError):
        TokenizerBase()  # type: ignore[abstract]


def test_mock_tokenizer_is_tokenizer_base():
    tok = MockTokenizer()
    assert isinstance(tok, TokenizerBase)
