"""
ModelWrapper: the main entry point.

Wraps any ModelInterface with:

  - Speculative decoding  (when a draft model is supplied)
  - Dynamic early exit    (when no draft model is supplied)
  - Token budget enforcement
  - GenerationMetrics collection
"""
from __future__ import annotations

from kairu.base import ModelInterface
from kairu.budget import TokenBudget
from kairu.early_exit import EarlyExitDecoder
from kairu.metrics import GenerationMetrics
from kairu.speculative import SpeculativeDecoder


class ModelWrapper:
    """
    Wraps a ModelInterface with optional optimization layers.

    Parameters
    ----------
    model:
        The primary (target) model.
    draft_model:
        Optional smaller draft model. When provided, speculative decoding
        is used. When absent, dynamic early exit is used instead.
    max_budget:
        Hard cap on prompt + generated tokens per call.
    speculative_gamma:
        Look-ahead window size for speculative decoding.
    early_exit_threshold:
        Top-prob threshold that triggers early exit (0 < t <= 1).
    temperature:
        Sampling temperature forwarded to both decoders.
    """

    def __init__(
        self,
        model: ModelInterface,
        draft_model: ModelInterface | None = None,
        max_budget: int = 512,
        speculative_gamma: int = 4,
        early_exit_threshold: float = 0.9,
        temperature: float = 1.0,
    ):
        self._model = model
        self._draft = draft_model
        self._max_budget = max_budget
        self._temperature = temperature

        if draft_model is not None:
            self._decoder: SpeculativeDecoder | EarlyExitDecoder = SpeculativeDecoder(
                target=model,
                draft=draft_model,
                gamma=speculative_gamma,
                temperature=temperature,
            )
        else:
            self._decoder = EarlyExitDecoder(
                model=model,
                confidence_threshold=early_exit_threshold,
                temperature=temperature,
            )

    @property
    def model(self) -> ModelInterface:
        return self._model

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
    ) -> tuple[list[int], GenerationMetrics]:
        """
        Generate tokens with budget enforcement and metrics tracking.

        Returns
        -------
        token_ids:
            Newly generated token ids (length <= min(max_new_tokens, budget.remaining)).
        metrics:
            Populated GenerationMetrics with timing and acceptance stats.
        """
        budget = TokenBudget(max_total_tokens=self._max_budget)
        budget.set_prompt(len(prompt_ids))
        effective_max = min(max_new_tokens, budget.remaining)

        metrics = GenerationMetrics(prompt_tokens=len(prompt_ids))

        generated, stats = self._decoder.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=effective_max,
        )

        metrics.generated_tokens = len(generated)
        if "accepted_tokens" in stats:
            metrics.accepted_tokens = stats["accepted_tokens"]
            metrics.rejected_tokens = stats["rejected_tokens"]

        metrics.finish()
        return generated, metrics


def wrap_model(
    model_or_name,
    draft_model: ModelInterface | None = None,
    max_budget: int = 512,
    **kwargs,
) -> ModelWrapper:
    """
    Convenience entry point.

    Accepts either a ModelInterface instance or a string model name.
    When given a string:
      - Attempts to load via HuggingFace (requires the ``hf`` extra).
      - Falls back to MockModel when HuggingFace is not installed.
    """
    if isinstance(model_or_name, str):
        try:
            from kairu._hf_backend import load_hf_model

            model: ModelInterface = load_hf_model(model_or_name)
        except Exception:  # noqa: BLE001 — covers ImportError + any HF load failure
            from kairu.mock_model import MockModel

            model = MockModel()
    else:
        model = model_or_name

    return ModelWrapper(model=model, draft_model=draft_model, max_budget=max_budget, **kwargs)
