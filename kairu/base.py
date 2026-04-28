from abc import ABC, abstractmethod
import numpy as np


class ModelInterface(ABC):
    """Minimal interface any model backend must satisfy."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        """Return logits over vocab for the next token. Shape: (vocab_size,)"""
        ...

    @abstractmethod
    def max_seq_len(self) -> int: ...
