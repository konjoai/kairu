"""Kairu — real-time LLM inference optimizer.

流 · to flow, to stream

Provides:
  - wrap_model   : main entry point; returns a ModelWrapper
  - ModelWrapper : wraps any ModelInterface with speculative decoding,
                   early exit, and token budget enforcement
  - TokenBudget  : hard per-generation token cap
  - GenerationMetrics : timing and acceptance-rate tracking
"""

__version__ = "0.1.0"

from kairu.budget import TokenBudget
from kairu.metrics import GenerationMetrics
from kairu.wrapper import ModelWrapper, wrap_model

__all__ = [
    "wrap_model",
    "ModelWrapper",
    "TokenBudget",
    "GenerationMetrics",
    "__version__",
]
