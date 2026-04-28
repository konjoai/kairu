import numpy as np
from kairu.base import ModelInterface


class EarlyExitDecoder:
    """
    Dynamic early exit: stop generating once the model is sufficiently
    confident — either the top-token probability exceeds confidence_threshold
    OR the distribution entropy drops below entropy_floor.

    Both conditions measure the same signal (certainty) from complementary
    angles: high max-prob → peaked distribution; low entropy → peaked distribution.
    """

    def __init__(
        self,
        model: ModelInterface,
        confidence_threshold: float = 0.9,
        entropy_floor: float = 0.5,
        temperature: float = 1.0,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.entropy_floor = entropy_floor
        self.temperature = temperature
        self._rng = np.random.default_rng(42)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _entropy(self, probs: np.ndarray) -> float:
        """Shannon entropy in nats. Clips to avoid log(0)."""
        p = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _should_exit(self, probs: np.ndarray) -> bool:
        top_prob = float(probs.max())
        entropy = self._entropy(probs)
        return top_prob >= self.confidence_threshold or entropy <= self.entropy_floor

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
    ) -> tuple[list[int], dict]:
        """
        Generate tokens, stopping early when the model is sufficiently confident.

        Returns:
            generated_ids: list of generated token ids
            stats: {tokens_generated, exit_reason, max_new_tokens, early_exit}
        """
        tokens = list(prompt_ids)
        generated: list[int] = []
        exit_reason = "max_tokens"

        for _ in range(max_new_tokens):
            logits = self.model.next_token_logits(tokens)
            probs = self._softmax(logits / max(self.temperature, 1e-8))

            if self._should_exit(probs):
                # Emit the argmax token (most confident choice), then halt
                tok = int(np.argmax(probs))
                generated.append(tok)
                tokens.append(tok)
                exit_reason = (
                    "confidence"
                    if float(probs.max()) >= self.confidence_threshold
                    else "entropy"
                )
                break

            tok = int(self._rng.choice(len(probs), p=probs))
            generated.append(tok)
            tokens.append(tok)

        stats = {
            "tokens_generated": len(generated),
            "exit_reason": exit_reason,
            "max_new_tokens": max_new_tokens,
            "early_exit": exit_reason != "max_tokens",
        }
        return generated, stats
