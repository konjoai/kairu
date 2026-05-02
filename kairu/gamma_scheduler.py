"""Dynamic gamma scheduler for speculative decoding.

The optimal lookahead γ depends on the draft/target acceptance rate ρ. From
Leviathan et al. (2023, "Fast Inference from Transformers via Speculative
Decoding"), the expected number of tokens accepted per verification round is

    E[accepted] = (1 - ρ^(γ+1)) / (1 - ρ)

so high ρ ⇒ a larger γ amortizes more target calls, while low ρ means most
draft tokens are wasted compute. We adapt γ online with an AIMD-style rule
over a sliding window of recent acceptance ratios:

    if rolling_acceptance ≥ high_threshold:  γ ← min(γ + increase, max_gamma)
    if rolling_acceptance ≤ low_threshold:   γ ← max(round(γ * decrease_factor), min_gamma)
    else:                                     γ unchanged

This is the same control law as TCP congestion control: additive increase on
success, multiplicative decrease on failure. It converges quickly without
oscillation when the underlying acceptance rate is stationary.
"""
from __future__ import annotations

from collections import deque


class DynamicGammaScheduler:
    """AIMD-style scheduler over speculative ``gamma``.

    Parameters
    ----------
    initial : int
        Starting γ. Must satisfy ``min_gamma ≤ initial ≤ max_gamma``.
    min_gamma, max_gamma : int
        Hard bounds; γ is clamped to ``[min_gamma, max_gamma]``.
    high_threshold, low_threshold : float
        Acceptance-rate thresholds for increase / decrease, both in (0, 1].
    increase : int
        Additive step on high acceptance.
    decrease_factor : float
        Multiplicative shrink factor in (0, 1] applied on low acceptance.
    window : int
        Number of recent rounds averaged into the rolling rate.
    """

    def __init__(
        self,
        initial: int = 4,
        min_gamma: int = 1,
        max_gamma: int = 16,
        high_threshold: float = 0.8,
        low_threshold: float = 0.4,
        increase: int = 1,
        decrease_factor: float = 0.5,
        window: int = 8,
    ) -> None:
        if not (1 <= min_gamma <= max_gamma):
            raise ValueError("require 1 <= min_gamma <= max_gamma")
        if not (min_gamma <= initial <= max_gamma):
            raise ValueError("initial must be within [min_gamma, max_gamma]")
        if not (0.0 < low_threshold < high_threshold <= 1.0):
            raise ValueError("require 0 < low_threshold < high_threshold <= 1")
        if not (0.0 < decrease_factor <= 1.0):
            raise ValueError("decrease_factor must be in (0, 1]")
        if increase < 1:
            raise ValueError("increase must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")

        self._gamma = initial
        self._min = min_gamma
        self._max = max_gamma
        self._high = high_threshold
        self._low = low_threshold
        self._inc = increase
        self._dec = decrease_factor
        self._history: deque[float] = deque(maxlen=window)
        self._adjustments = 0

    @property
    def gamma(self) -> int:
        return self._gamma

    def rolling_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def update(self, accepted: int, attempted: int) -> int:
        """Record one verification round and return the new γ.

        ``accepted`` is the number of draft tokens accepted; ``attempted`` is
        γ used for that round (always >= 1 in practice).
        """
        if attempted < 1:
            raise ValueError("attempted must be >= 1")
        if accepted < 0 or accepted > attempted:
            raise ValueError("accepted must be in [0, attempted]")

        rate = accepted / attempted
        self._history.append(rate)
        smoothed = self.rolling_rate()

        prev = self._gamma
        if smoothed >= self._high:
            self._gamma = min(self._gamma + self._inc, self._max)
        elif smoothed <= self._low:
            self._gamma = max(int(round(self._gamma * self._dec)), self._min)
        if self._gamma != prev:
            self._adjustments += 1
        return self._gamma

    def stats(self) -> dict:
        return {
            "gamma": self._gamma,
            "rolling_acceptance": self.rolling_rate(),
            "adjustments": self._adjustments,
            "min_gamma": self._min,
            "max_gamma": self._max,
            "window_size": len(self._history),
        }


__all__ = ["DynamicGammaScheduler"]
