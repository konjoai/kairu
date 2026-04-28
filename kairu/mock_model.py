import numpy as np
from kairu.base import ModelInterface


class MockModel(ModelInterface):
    """
    Deterministic mock model for testing. Uses LCG-seeded logits based on
    token context so same prefix always yields the same logit distribution.
    No ML framework required.
    """

    VOCAB_SIZE = 1000
    MAX_SEQ = 512

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def max_seq_len(self) -> int:
        return self.MAX_SEQ

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        # LCG-style seed: Knuth multiplicative hash of token sum
        seed = (sum(token_ids) * 2654435761) % (2**32) if token_ids else 42
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(self.VOCAB_SIZE).astype(np.float32)
        # Add bias to a preferred token to simulate non-uniform distribution
        preferred = (sum(token_ids) * 7 + 13) % self.VOCAB_SIZE if token_ids else 0
        logits[preferred] += 3.0
        return logits
