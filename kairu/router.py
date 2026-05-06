"""Automatic decoder strategy router based on prompt analysis and runtime signals."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .base import ModelInterface
from .auto_profile import AutoProfile, DecoderProfile
from .metrics import GenerationMetrics


@dataclass
class RouterDecision:
    """Outcome of a routing decision."""

    strategy: str  # "speculative" | "early_exit" | "streaming"
    profile: DecoderProfile
    confidence: float  # 0.0-1.0
    rationale: str
    latency_budget_ms: Optional[float] = None


@dataclass
class RoutingStats:
    """Accumulated routing statistics for feedback."""

    decisions: Dict[str, int] = field(
        default_factory=lambda: {"speculative": 0, "early_exit": 0, "streaming": 0}
    )
    mean_latency_by_strategy: Dict[str, float] = field(default_factory=dict)
    total_routed: int = 0


class DecoderRouter:
    """Route prompts to the optimal decoder strategy.

    Routing logic:
    - Short prompts (< ``short_prompt_threshold`` tokens) → streaming
      (low latency, no draft model needed)
    - Long prompts with draft model available → speculative
      (highest throughput)
    - Long prompts, no draft model → early_exit
    - Latency budget set and tight (< 200 ms) → streaming fallback
      regardless of prompt length
    """

    def __init__(
        self,
        model: ModelInterface,
        draft_model: Optional[ModelInterface] = None,
        short_prompt_threshold: int = 20,
        latency_budget_ms: Optional[float] = None,
        name_hint: str = "",
    ) -> None:
        if short_prompt_threshold < 1:
            raise ValueError("short_prompt_threshold must be >= 1")
        self._model = model
        self._draft = draft_model
        self._threshold = short_prompt_threshold
        self._budget_ms = latency_budget_ms
        self._name = name_hint
        self._stats = RoutingStats()

    @property
    def stats(self) -> RoutingStats:
        return self._stats

    def route(self, prompt_token_ids: List[int]) -> RouterDecision:
        """Select the best decoding strategy for the given prompt token ids."""
        n_tokens = len(prompt_token_ids)
        has_draft = self._draft is not None
        budget_tight = (
            self._budget_ms is not None and self._budget_ms < 200.0
        )

        # Budget-first override: tight latency budget → streaming
        if budget_tight:
            strategy = "streaming"
            rationale = (
                f"Latency budget {self._budget_ms:.0f}ms is tight; "
                "streaming minimises TTFT."
            )
            confidence = 0.9
        elif n_tokens < self._threshold:
            strategy = "streaming"
            rationale = (
                f"Short prompt ({n_tokens} tokens < threshold {self._threshold}); "
                "streaming preferred."
            )
            confidence = 0.85
        elif has_draft:
            strategy = "speculative"
            rationale = (
                f"Draft model available; speculative decoding expected to maximise "
                f"throughput on {n_tokens}-token prompt."
            )
            confidence = 0.9
        else:
            strategy = "early_exit"
            rationale = (
                f"No draft model; early-exit with confidence threshold on "
                f"{n_tokens}-token prompt."
            )
            confidence = 0.75

        profile = AutoProfile.recommend(
            self._model,
            name_hint=self._name,
            has_draft=has_draft,
        )
        # Override the profile strategy to match routing decision
        profile = dataclasses.replace(profile, strategy=strategy)

        self._stats.decisions[strategy] = self._stats.decisions.get(strategy, 0) + 1
        self._stats.total_routed += 1

        return RouterDecision(
            strategy=strategy,
            profile=profile,
            confidence=confidence,
            rationale=rationale,
            latency_budget_ms=self._budget_ms,
        )

    def record_outcome(
        self, decision: RouterDecision, metrics: GenerationMetrics
    ) -> None:
        """Feed back actual latency to refine future routing (EWMA)."""
        s = decision.strategy
        # Derive wall-clock seconds from GenerationMetrics.total_time_ms
        latency_ms = metrics.total_time_ms
        prev = self._stats.mean_latency_by_strategy.get(s)
        if prev is None:
            self._stats.mean_latency_by_strategy[s] = latency_ms
        else:
            alpha = 0.2
            self._stats.mean_latency_by_strategy[s] = (
                alpha * latency_ms + (1 - alpha) * prev
            )


__all__ = ["DecoderRouter", "RouterDecision", "RoutingStats"]
