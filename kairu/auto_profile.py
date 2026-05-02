"""AutoProfile — pick a decoder strategy from model metadata.

The decision surface is small and stable enough to encode as a deterministic
heuristic rather than a learned classifier:

  * **vanilla** — small/draft models. Speculative decoding has no smaller draft
    to lean on; early exit on a tiny network would fire on every token.
  * **early_exit** — mid-size models with a single backbone. The mean exit
    layer typically lands around 60–70 % of depth for non-trivial prompts.
  * **layered_early_exit** — same as above but the model exposes per-layer
    logits via :class:`LayeredModelInterface`.
  * **speculative** — large target models (vocab > 30k, or family pattern
    matches a known frontier family). When a draft model is supplied, this is
    almost always the right choice; γ scales with the size ratio.

Every case enables the logits cache (``use_cache=True``) by default —
memoization is a strict win whenever the workload has any prefix overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from kairu.base import ModelInterface

_FRONTIER_FAMILIES = re.compile(
    r"\b(llama|mistral|mixtral|qwen|deepseek|phi|gpt|gemma|claude)\b",
    re.IGNORECASE,
)
_DRAFT_HINTS = re.compile(r"\b(tiny|small|mini|draft|125m|350m)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DecoderProfile:
    """Recommended decoder configuration."""

    strategy: str  # "vanilla" | "early_exit" | "layered_early_exit" | "speculative"
    gamma: int
    early_exit_threshold: float
    temperature: float
    use_cache: bool
    cache_capacity: int
    rationale: str


class AutoProfile:
    """Recommend a :class:`DecoderProfile` from a model + optional name hint."""

    @staticmethod
    def recommend(
        model: ModelInterface,
        name_hint: str | None = None,
        has_draft: bool = False,
    ) -> DecoderProfile:
        from kairu.layered import LayeredModelInterface  # local — avoid cycle

        vocab = int(model.vocab_size)
        is_layered = isinstance(model, LayeredModelInterface)
        name = name_hint or ""

        if name and _DRAFT_HINTS.search(name):
            return DecoderProfile(
                strategy="vanilla",
                gamma=1,
                early_exit_threshold=1.0,
                temperature=1.0,
                use_cache=True,
                cache_capacity=64,
                rationale=f"name '{name}' matches draft-model pattern; no speculation, no early exit",
            )

        if has_draft and (vocab >= 30_000 or _FRONTIER_FAMILIES.search(name)):
            gamma = 6 if vocab >= 100_000 else 4
            return DecoderProfile(
                strategy="speculative",
                gamma=gamma,
                early_exit_threshold=0.9,
                temperature=1.0,
                use_cache=True,
                cache_capacity=256,
                rationale=(
                    f"large target (vocab={vocab})"
                    f"{' / frontier family' if _FRONTIER_FAMILIES.search(name) else ''}"
                    f" with draft → speculative γ={gamma}"
                ),
            )

        if is_layered:
            return DecoderProfile(
                strategy="layered_early_exit",
                gamma=1,
                early_exit_threshold=0.85,
                temperature=1.0,
                use_cache=True,
                cache_capacity=128,
                rationale="model exposes per-layer logits → architecture-aware early exit",
            )

        if vocab >= 5_000:
            return DecoderProfile(
                strategy="early_exit",
                gamma=1,
                early_exit_threshold=0.9,
                temperature=1.0,
                use_cache=True,
                cache_capacity=128,
                rationale=f"mid-size single-backbone model (vocab={vocab}) → confidence early exit",
            )

        return DecoderProfile(
            strategy="vanilla",
            gamma=1,
            early_exit_threshold=1.0,
            temperature=1.0,
            use_cache=True,
            cache_capacity=32,
            rationale=f"tiny model (vocab={vocab}) → vanilla decoding",
        )


__all__ = ["AutoProfile", "DecoderProfile"]
