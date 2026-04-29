"""Kairu — real-time LLM inference optimizer.

流 · to flow, to stream

Provides:
  - wrap_model   : main entry point; returns a ModelWrapper
  - ModelWrapper : wraps any ModelInterface with speculative decoding,
                   early exit, and token budget enforcement
  - TokenBudget  : hard per-generation token cap
  - GenerationMetrics : timing and acceptance-rate tracking
  - StreamingDecoder  : token-by-token streaming generation
  - MockTokenizer     : deterministic mock tokenizer for testing
  - TokenizerBase     : abstract tokenizer interface
"""
from __future__ import annotations

__version__ = "0.3.0"

from kairu.bench import BenchmarkResult, BenchmarkRunner
from kairu.budget import TokenBudget
from kairu.metrics import GenerationMetrics
from kairu.streaming import StreamingDecoder
from kairu.tokenizer import MockTokenizer, TokenizerBase
from kairu.wrapper import ModelWrapper, wrap_model

try:
    from kairu.tokenizer import HFTokenizer
except Exception:  # noqa: BLE001 — transformers not installed
    HFTokenizer = None  # type: ignore[assignment,misc]

__all__ = [
    "wrap_model",
    "ModelWrapper",
    "TokenBudget",
    "GenerationMetrics",
    "StreamingDecoder",
    "MockTokenizer",
    "TokenizerBase",
    "HFTokenizer",
    "BenchmarkRunner",
    "BenchmarkResult",
    "__version__",
]
