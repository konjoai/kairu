"""Streaming generation: yields tokens one by one via an iterator."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from kairu.base import ModelInterface


class StreamingDecoder:
    """
    Greedy or temperature-sampled streaming decoder.
    Yields token IDs one at a time; call list() to collect all.
    """

    def __init__(self, model: ModelInterface, temperature: float = 1.0) -> None:
        self._model = model
        self._temperature = temperature
        self._rng = np.random.default_rng(42)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _sample(self, logits: np.ndarray) -> int:
        if self._temperature == 0.0:
            return int(np.argmax(logits))
        probs = self._softmax(logits / max(self._temperature, 1e-8))
        return int(self._rng.choice(len(probs), p=probs))

    def stream(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        stop_token_id: int | None = None,
    ) -> Iterator[int]:
        """Yield token IDs one at a time."""
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            logits = self._model.next_token_logits(tokens)
            tok = self._sample(logits)
            yield tok
            tokens.append(tok)
            if stop_token_id is not None and tok == stop_token_id:
                break

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 50) -> list[int]:
        """Collect all streamed tokens into a list."""
        return list(self.stream(prompt_ids, max_new_tokens))
