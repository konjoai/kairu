import numpy as np
from kairu.base import ModelInterface


class SpeculativeDecoder:
    """
    Speculative decoding: a small draft model generates candidate tokens,
    the target model verifies them in parallel. Accepted tokens are kept;
    rejected tokens cause a fallback to target-model sampling.

    Reference: Chen et al. (2023) "Accelerating Large Language Model Decoding
    with Speculative Sampling" — https://arxiv.org/abs/2302.01318
    """

    def __init__(
        self,
        target: ModelInterface,
        draft: ModelInterface,
        gamma: int = 4,
        temperature: float = 1.0,
    ):
        self.target = target
        self.draft = draft
        self.gamma = gamma
        self.temperature = temperature
        self._rng = np.random.default_rng(42)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _sample(self, logits: np.ndarray) -> int:
        """Temperature sampling from logits."""
        if self.temperature == 0.0:
            return int(np.argmax(logits))
        probs = self._softmax(logits / self.temperature)
        return int(self._rng.choice(len(probs), p=probs))

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
    ) -> tuple[list[int], dict]:
        """
        Generate tokens using speculative decoding.

        Returns:
            generated_token_ids: list of new token ids (length <= max_new_tokens)
            stats: {total_tokens, accepted_tokens, rejected_tokens, acceptance_rate}
        """
        if max_new_tokens == 0:
            return [], {
                "total_tokens": 0,
                "accepted_tokens": 0,
                "rejected_tokens": 0,
                "acceptance_rate": 0.0,
            }

        tokens = list(prompt_ids)
        generated: list[int] = []
        total_accepted = 0
        total_rejected = 0

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            gamma = min(self.gamma, remaining)

            # --- Draft phase: generate gamma candidate tokens ---
            draft_tokens: list[int] = []
            draft_probs: list[float] = []
            draft_ctx = list(tokens)
            for _ in range(gamma):
                logits = self.draft.next_token_logits(draft_ctx)
                probs = self._softmax(logits / max(self.temperature, 1e-8))
                tok = int(self._rng.choice(len(probs), p=probs))
                draft_tokens.append(tok)
                draft_probs.append(float(probs[tok]))
                draft_ctx.append(tok)

            # --- Verify phase: target model accepts or rejects each draft token ---
            accepted: list[int] = []
            verify_ctx = list(tokens)
            all_accepted = True
            for dtok, dprob in zip(draft_tokens, draft_probs):
                target_logits = self.target.next_token_logits(verify_ctx)
                target_probs = self._softmax(target_logits / max(self.temperature, 1e-8))
                accept_ratio = min(1.0, float(target_probs[dtok]) / max(dprob, 1e-10))

                if self._rng.random() < accept_ratio:
                    accepted.append(dtok)
                    verify_ctx.append(dtok)
                    total_accepted += 1
                else:
                    total_rejected += 1
                    all_accepted = False
                    # Sample from the adjusted (residual) distribution
                    adj = np.maximum(
                        target_probs - self._softmax(target_logits / max(self.temperature, 1e-8)),
                        0.0,
                    )
                    adj_sum = float(adj.sum())
                    if adj_sum > 1e-10:
                        adj /= adj_sum
                        fallback = int(self._rng.choice(len(adj), p=adj))
                    else:
                        fallback = self._sample(target_logits)
                    accepted.append(fallback)
                    break

            tokens.extend(accepted)
            generated.extend(accepted)

            if len(generated) >= max_new_tokens:
                break

            # If all gamma draft tokens were accepted, sample one bonus token from target
            if all_accepted and len(accepted) == gamma:
                target_logits = self.target.next_token_logits(tokens)
                bonus = self._sample(target_logits)
                tokens.append(bonus)
                generated.append(bonus)
                total_accepted += 1

        generated = generated[:max_new_tokens]
        total = total_accepted + total_rejected
        stats = {
            "total_tokens": len(generated),
            "accepted_tokens": total_accepted,
            "rejected_tokens": total_rejected,
            "acceptance_rate": total_accepted / total if total > 0 else 0.0,
        }
        return generated, stats
