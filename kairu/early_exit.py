"""Dynamic early-exit decoding — halt generation once the model is confident.

Two threshold regimes are available:

* **Static** (default) — a fixed ``confidence_threshold`` and ``entropy_floor``,
  identical to prior releases.
* **Adaptive** (``adaptive=True``) — a CALM-style per-token *decaying* confidence
  threshold (Schuster et al. 2022, "Confident Adaptive Language Modeling").
  Early tokens must clear a high bar to trigger an exit (an early mistake
  propagates through every later token), while the bar relaxes geometrically
  toward ``min_confidence`` as decoding proceeds::

      threshold(t) = min_confidence
                   + (confidence_threshold - min_confidence) * exp(-decay * t)

  At ``t = 0`` the threshold equals ``confidence_threshold``; as ``t → ∞`` it
  approaches ``min_confidence``. The entropy floor is unchanged — confidence is
  the adaptive signal, entropy remains the complementary static guard.
"""

from __future__ import annotations

import math

import numpy as np

from kairu.base import ModelInterface


class EarlyExitDecoder:
    """Stop generating once the model is sufficiently confident.

    An exit fires when the top-token probability clears the (optionally
    adaptive) confidence threshold OR the distribution entropy drops below
    ``entropy_floor`` — both measure certainty from complementary angles.
    """

    def __init__(
        self,
        model: ModelInterface,
        confidence_threshold: float = 0.9,
        entropy_floor: float = 0.5,
        temperature: float = 1.0,
        *,
        adaptive: bool = False,
        min_confidence: float = 0.5,
        adapt_decay: float = 0.2,
    ):
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if entropy_floor < 0.0:
            raise ValueError("entropy_floor must be >= 0")
        if adapt_decay < 0.0:
            raise ValueError("adapt_decay must be >= 0")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.entropy_floor = entropy_floor
        self.temperature = temperature
        self.adaptive = adaptive
        # The relaxed floor can never exceed the base bar, or the schedule would
        # rise instead of decay — clamp so it stays monotone non-increasing.
        self.min_confidence = min(min_confidence, confidence_threshold)
        self.adapt_decay = adapt_decay
        self._rng = np.random.default_rng(42)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _entropy(self, probs: np.ndarray) -> float:
        """Shannon entropy in nats. Clips to avoid log(0)."""
        p = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))

    def effective_confidence(self, step: int) -> float:
        """The confidence threshold in force at generation ``step`` (0-based).

        Constant in static mode; a decaying geometric schedule from
        ``confidence_threshold`` down toward ``min_confidence`` when adaptive.
        """
        if not self.adaptive:
            return self.confidence_threshold
        span = self.confidence_threshold - self.min_confidence
        return self.min_confidence + span * math.exp(-self.adapt_decay * step)

    def _should_exit(self, probs: np.ndarray, step: int) -> tuple[bool, str]:
        """Return ``(exit_now, reason)`` for the distribution at ``step``."""
        if float(probs.max()) >= self.effective_confidence(step):
            return True, "confidence"
        if self._entropy(probs) <= self.entropy_floor:
            return True, "entropy"
        return False, ""

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
    ) -> tuple[list[int], dict]:
        """Generate tokens, halting early once the model is confident enough.

        Returns ``(generated_ids, stats)`` where ``stats`` carries
        ``tokens_generated``, ``exit_reason``, ``max_new_tokens``,
        ``early_exit``, ``adaptive``, and ``final_confidence_threshold`` (the
        effective threshold at the step where generation stopped).
        """
        tokens = list(prompt_ids)
        generated: list[int] = []
        exit_reason = "max_tokens"
        threshold = self.confidence_threshold

        for step in range(max_new_tokens):
            logits = self.model.next_token_logits(tokens)
            probs = self._softmax(logits / max(self.temperature, 1e-8))
            threshold = self.effective_confidence(step)

            exit_now, reason = self._should_exit(probs, step)
            if exit_now:
                # Emit the argmax token (most confident choice), then halt.
                tok = int(np.argmax(probs))
                generated.append(tok)
                tokens.append(tok)
                exit_reason = reason
                break

            tok = int(self._rng.choice(len(probs), p=probs))
            generated.append(tok)
            tokens.append(tok)

        stats = {
            "tokens_generated": len(generated),
            "exit_reason": exit_reason,
            "max_new_tokens": max_new_tokens,
            "early_exit": exit_reason != "max_tokens",
            "adaptive": self.adaptive,
            "final_confidence_threshold": threshold,
        }
        return generated, stats


__all__ = ["EarlyExitDecoder"]
