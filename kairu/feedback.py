"""Feedback loop: wire benchmark results into the gamma scheduler."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List

from .bench import BenchmarkResult
from .gamma_scheduler import DynamicGammaScheduler

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSummary:
    """Summary emitted after each feedback cycle."""

    n_results: int
    mean_acceptance_rate: float
    gamma_adjusted: bool
    new_gamma: Optional[float]
    recommendation: str


class FeedbackLoop:
    """Collect BenchmarkResults and adjust a DynamicGammaScheduler accordingly.

    High acceptance rate → try larger gamma (more speculative steps).
    Low acceptance rate  → shrink gamma (fewer wasted draft steps).
    """

    HIGH_ACCEPTANCE_THRESHOLD = 0.75
    LOW_ACCEPTANCE_THRESHOLD = 0.40

    def __init__(
        self, scheduler: DynamicGammaScheduler, min_results: int = 5
    ) -> None:
        if min_results < 1:
            raise ValueError("min_results must be >= 1")
        self._scheduler = scheduler
        self._min_results = min_results
        self._buffer: List[BenchmarkResult] = []

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def ingest(self, result: BenchmarkResult) -> Optional[FeedbackSummary]:
        """Add a BenchmarkResult. Returns a FeedbackSummary once enough data is buffered."""
        self._buffer.append(result)
        if len(self._buffer) < self._min_results:
            return None
        return self._flush()

    def _flush(self) -> FeedbackSummary:
        results = self._buffer[:]
        self._buffer.clear()

        # Collect acceptance rates from result metadata dicts
        acceptance_rates = [
            r.metadata.get("acceptance_rate", None)
            for r in results
            if isinstance(r.metadata, dict)
        ]
        valid = [a for a in acceptance_rates if a is not None]
        mean_ar = sum(valid) / len(valid) if valid else 0.5

        gamma_adjusted = False
        new_gamma: Optional[float] = None
        recommendation = "No adjustment — insufficient acceptance-rate data."

        if valid:
            current_gamma = self._scheduler.gamma
            if mean_ar > self.HIGH_ACCEPTANCE_THRESHOLD:
                # Positive signal: drive gamma up via high-acceptance updates
                # update() needs accepted & attempted integers; use gamma as attempted
                gamma_val = self._scheduler.gamma
                for _ in range(3):
                    self._scheduler.update(
                        accepted=gamma_val, attempted=gamma_val
                    )  # 100 % acceptance → will increase gamma
                gamma_adjusted = True
                new_gamma = float(self._scheduler.gamma)
                recommendation = (
                    f"High acceptance rate ({mean_ar:.2f}); "
                    f"increased gamma to {new_gamma:.2f}."
                )
            elif mean_ar < self.LOW_ACCEPTANCE_THRESHOLD:
                gamma_val = self._scheduler.gamma
                for _ in range(3):
                    self._scheduler.update(
                        accepted=0, attempted=gamma_val
                    )  # 0 % acceptance → will decrease gamma
                gamma_adjusted = True
                new_gamma = float(self._scheduler.gamma)
                recommendation = (
                    f"Low acceptance rate ({mean_ar:.2f}); "
                    f"decreased gamma to {new_gamma:.2f}."
                )
            else:
                recommendation = (
                    f"Acceptance rate ({mean_ar:.2f}) within normal range; "
                    "no gamma change."
                )

        if gamma_adjusted:
            logger.info("FeedbackLoop: %s", recommendation)

        return FeedbackSummary(
            n_results=len(results),
            mean_acceptance_rate=mean_ar,
            gamma_adjusted=gamma_adjusted,
            new_gamma=new_gamma,
            recommendation=recommendation,
        )


__all__ = ["FeedbackLoop", "FeedbackSummary"]
