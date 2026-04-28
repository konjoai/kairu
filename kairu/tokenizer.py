"""Tokenizer abstraction — wraps any HF tokenizer or works standalone as a mock."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TokenizerBase(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str: ...

    @abstractmethod
    def vocab_size(self) -> int: ...


class MockTokenizer(TokenizerBase):
    """Deterministic mock tokenizer for testing. Space-splits, hashes words to token IDs."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self._vocab = vocab_size

    def encode(self, text: str) -> list[int]:
        return [hash(w) % self._vocab for w in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"<{i}>" for i in token_ids)

    def vocab_size(self) -> int:
        return self._vocab


class HFTokenizer(TokenizerBase):
    """Wraps a HuggingFace AutoTokenizer."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def vocab_size(self) -> int:
        return self._tok.vocab_size
