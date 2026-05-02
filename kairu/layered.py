"""Architecture-aware early exit — halt at intermediate transformer layers.

When a model exposes per-layer next-token logits, we can avoid computing the
remaining layers once the distribution is already peaked. The compute saved is

    saved = (L - L_exit) / L

per token, where ``L`` is the total layer count and ``L_exit`` is the layer at
which the confidence threshold first fired.

The :class:`LayeredModelInterface` extension is opt-in. Real HF transformers
expose intermediate hidden states via ``output_hidden_states=True``; the
:class:`MockLayeredModel` here provides a deterministic, dependency-free
implementation that progressively sharpens its logits as layer index grows
— the asymptotic behavior of a well-trained transformer.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from kairu.base import ModelInterface


class LayeredModelInterface(ModelInterface):
    """Extension exposing per-layer logits.

    ``layer_logits(tokens, layer_idx)`` returns the logits *as if the forward
    pass had stopped after layer ``layer_idx``* — typically the projection of
    layer ``layer_idx``'s hidden state through the LM head (see DeeBERT,
    LayerSkip — Elhoushi et al. 2024).
    """

    @abstractmethod
    def num_layers(self) -> int:
        """Total number of layers in the model. Must be >= 1."""

    @abstractmethod
    def layer_logits(self, token_ids: list[int], layer_idx: int) -> np.ndarray:
        """Logits if the forward pass halted at ``layer_idx`` (1-indexed, ≤ num_layers)."""


class MockLayeredModel(LayeredModelInterface):
    """Deterministic L-layer mock — distribution sharpens monotonically with depth.

    Math: ``logits(layer_l) = base_logits * (l / L)`` where ``base_logits`` is
    the LCG-seeded distribution from :class:`MockModel`. As ``l → L`` the
    softmax becomes more peaked, simulating the empirical observation that
    deeper transformer layers produce more confident next-token distributions.
    """

    VOCAB_SIZE = 1000
    MAX_SEQ = 512

    def __init__(self, num_layers: int = 12) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self._L = num_layers

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def max_seq_len(self) -> int:
        return self.MAX_SEQ

    def num_layers(self) -> int:
        return self._L

    def _base_logits(self, token_ids: list[int]) -> np.ndarray:
        seed = (sum(token_ids) * 2654435761) % (2**32) if token_ids else 42
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(self.VOCAB_SIZE).astype(np.float32)
        preferred = (sum(token_ids) * 7 + 13) % self.VOCAB_SIZE if token_ids else 0
        logits[preferred] += 3.0
        return logits

    def layer_logits(self, token_ids: list[int], layer_idx: int) -> np.ndarray:
        if not 1 <= layer_idx <= self._L:
            raise ValueError(f"layer_idx must be in [1, {self._L}], got {layer_idx}")
        scale = layer_idx / self._L
        return (self._base_logits(token_ids) * scale).astype(np.float32)

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        return self.layer_logits(token_ids, self._L)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class LayerwiseEarlyExitDecoder:
    """Per-layer early exit. Walks layers ``[min_layer, num_layers]`` and emits
    the argmax token at the first layer whose top-prob ≥ ``confidence_threshold``.

    Per-token output stats include the exit layer and the fraction of total
    layers skipped — the compute saved is exactly that fraction (assuming
    uniform per-layer cost, which is true for transformer decoder stacks).
    """

    def __init__(
        self,
        model: LayeredModelInterface,
        confidence_threshold: float = 0.9,
        min_layer: int = 1,
        temperature: float = 1.0,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if min_layer < 1:
            raise ValueError("min_layer must be >= 1")
        self._model = model
        self._threshold = confidence_threshold
        self._min_layer = min_layer
        self._temperature = max(temperature, 1e-8)

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int = 50
    ) -> tuple[list[int], dict]:
        L = self._model.num_layers()
        if self._min_layer > L:
            raise ValueError(f"min_layer ({self._min_layer}) > num_layers ({L})")

        tokens = list(prompt_ids)
        generated: list[int] = []
        exit_layers: list[int] = []

        for _ in range(max_new_tokens):
            chosen_layer = L
            chosen_tok = 0
            for layer in range(self._min_layer, L + 1):
                logits = self._model.layer_logits(tokens, layer) / self._temperature
                probs = _softmax(logits)
                if float(probs.max()) >= self._threshold or layer == L:
                    chosen_layer = layer
                    chosen_tok = int(np.argmax(probs))
                    break
            generated.append(chosen_tok)
            tokens.append(chosen_tok)
            exit_layers.append(chosen_layer)

        mean_exit = sum(exit_layers) / len(exit_layers) if exit_layers else float(L)
        compute_saved = 1.0 - (mean_exit / L)
        stats = {
            "tokens_generated": len(generated),
            "exit_layers": exit_layers,
            "mean_exit_layer": mean_exit,
            "total_layers": L,
            "compute_saved": compute_saved,
        }
        return generated, stats


__all__ = [
    "LayeredModelInterface",
    "MockLayeredModel",
    "LayerwiseEarlyExitDecoder",
]
