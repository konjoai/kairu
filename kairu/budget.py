from dataclasses import dataclass, field


@dataclass
class TokenBudget:
    """
    Enforces a hard token budget per generation.

    Tracks prompt tokens + generated tokens against a total cap.
    The budget is one-way: once tokens are consumed they cannot be
    un-consumed (use reset_generated() to restart a generation).
    """

    max_total_tokens: int
    _prompt_tokens: int = field(default=0, init=False)
    _generated_tokens: int = field(default=0, init=False)

    def set_prompt(self, n_tokens: int) -> None:
        """Record the number of prompt tokens. Must be called before generation."""
        if n_tokens < 0:
            raise ValueError(f"Prompt tokens cannot be negative, got {n_tokens}")
        self._prompt_tokens = n_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._generated_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_total_tokens - self.total_tokens)

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def consume(self, n: int = 1) -> int:
        """
        Consume n tokens from the budget.
        Returns the number of tokens actually consumed (<= n).
        When the budget is exhausted, returns 0.
        """
        allowed = min(n, self.remaining)
        self._generated_tokens += allowed
        return allowed

    def reset_generated(self) -> None:
        """Reset the generated-token counter, preserving the prompt count."""
        self._generated_tokens = 0

    def utilization(self) -> float:
        """Fraction of the total budget consumed (0.0 to 1.0)."""
        if self.max_total_tokens == 0:
            return 1.0
        return self.total_tokens / self.max_total_tokens
