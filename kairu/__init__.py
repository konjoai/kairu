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
  - DecoderRouter     : automatic strategy router
  - FeedbackLoop      : benchmark-driven gamma scheduler feedback
  - WatermarkLogitsProcessor : green/red list logit bias (Kirchenbauer 2023)
  - WatermarkDetector        : z-score watermark detector
"""
from __future__ import annotations

__version__ = "0.9.0"

from kairu.auto_profile import AutoProfile, DecoderProfile
from kairu.bench import BenchmarkResult, BenchmarkRunner
from kairu.budget import TokenBudget
from kairu.cluster_budget import ClusterTokenBudget, LocalClusterBudget
from kairu.feedback import FeedbackLoop, FeedbackSummary
from kairu.gamma_scheduler import DynamicGammaScheduler
from kairu.kv_cache import CachedModel, LogitsCache
from kairu.layered import (
    LayeredModelInterface,
    LayerwiseEarlyExitDecoder,
    MockLayeredModel,
)
from kairu.metrics import GenerationMetrics
from kairu.router import DecoderRouter, RouterDecision, RoutingStats
from kairu.streaming import StreamingDecoder
from kairu.tokenizer import MockTokenizer, TokenizerBase
from kairu.tracing import KairuTracer, extract_trace_context
from kairu.watermark import WatermarkDetector, WatermarkLogitsProcessor, WatermarkResult
from kairu.wrapper import ModelWrapper, wrap_model

try:
    from kairu.tokenizer import HFTokenizer
except Exception:  # noqa: BLE001 — transformers not installed
    HFTokenizer = None  # type: ignore[assignment,misc]

from kairu.metrics_export import MetricsCollector
from kairu.rate_limit import (
    InMemoryBackend,
    RateLimiter,
    RateLimiterBackend,
    RedisBackend,
)

try:
    from kairu.server import ServerConfig, create_app
except Exception:  # noqa: BLE001 — fastapi not installed
    create_app = None  # type: ignore[assignment]
    ServerConfig = None  # type: ignore[assignment,misc]

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
    "create_app",
    "ServerConfig",
    "RateLimiter",
    "RateLimiterBackend",
    "InMemoryBackend",
    "RedisBackend",
    "MetricsCollector",
    "AutoProfile",
    "DecoderProfile",
    "CachedModel",
    "LogitsCache",
    "DynamicGammaScheduler",
    "LayeredModelInterface",
    "LayerwiseEarlyExitDecoder",
    "MockLayeredModel",
    "KairuTracer",
    "extract_trace_context",
    "ClusterTokenBudget",
    "LocalClusterBudget",
    "DecoderRouter",
    "RouterDecision",
    "RoutingStats",
    "FeedbackLoop",
    "FeedbackSummary",
    "WatermarkLogitsProcessor",
    "WatermarkDetector",
    "WatermarkResult",
    "__version__",
]
