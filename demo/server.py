"""Kairu demo server — serves the dark-theme demo page and exposes endpoints
that drive every interactive widget from the *real* library code.

Run from the repo root::

    python demo/server.py            # listens on :7777
    python demo/server.py --port 9000

Endpoints
---------
``GET  /``                  → ``demo/index.html``
``GET  /api/health``        → ``{ok, version}``
``POST /api/speedup``       → real expected speedup from the closed-form
                              formula derived from Leviathan et al. 2023.
``POST /api/recommend``     → real :class:`AutoProfile.recommend` invocation
                              against a model constructed from the requested
                              spec (vocab size + draft availability + layered
                              architecture flag).
``POST /api/simulate-race`` → seeded-RNG simulation of speculative decoding,
                              token-by-token, against a real
                              :class:`DynamicGammaScheduler`.
``POST /api/kv-cache-sim``  → LogitsCache LRU simulation with per-op trace.
``POST /api/early-exit-sim`` → LayerwiseEarlyExitDecoder simulation.
``POST /api/watermark-demo`` → watermark apply + detect via Kirchenbauer 2023.

All three POST endpoints return JSON. CORS is open (this is a local demo).
Pure stdlib HTTP — no FastAPI / uvicorn / aiohttp dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

# Make `kairu` importable when launched from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairu import (  # noqa: E402 — sys.path manipulation must precede
    AutoProfile,
    DynamicGammaScheduler,
    LogitsCache,
    MockLayeredModel,
    __version__ as KAIRU_VERSION,
)
from kairu.evaluation import evaluate as eval_one  # noqa: E402
from kairu.mock_model import MockModel  # noqa: E402
from kairu.rubrics import RUBRIC_DEFS, rubric_names  # noqa: E402
from kairu.layered import LayerwiseEarlyExitDecoder  # noqa: E402
from kairu.watermark import WatermarkDetector, WatermarkLogitsProcessor  # noqa: E402
from kairu.streaming import StreamingDecoder  # noqa: E402

# Prism input limits — boundary validation per .claude/rules/security.md.
PRISM_MAX_TEXT = 16_384

logger = logging.getLogger("kairu.demo")

INDEX_PATH = Path(__file__).resolve().parent / "index.html"


# ════════════════════════════════════════════════════════════════════════════
# Math: the closed-form expected speedup for speculative decoding.
#
# Per the demo spec the chosen formulation is
#
#     E[speedup](ρ, γ) = (1 - ρ^(γ+1)) / ((1 - ρ) * (1 + γ * (1 - ρ)))
#
# This is a draft-cost-adjusted variant: the numerator is the expected number
# of tokens accepted in one verification round (Leviathan et al. 2023, Thm 3.8
# numerator), and the denominator multiplies the per-target cost by an effective
# work factor that grows with γ when ρ is low (γ extra draft calls earn little).
# At ρ → 1 it limits to (γ+1)/(1) and at ρ → 0 it limits to 1/(1+γ) — both
# correct intuitions: perfect draft fully amortizes the target call, useless
# draft pays γ wasted calls per accepted token.
#
# The expected number of tokens generated per verification round is the
# Leviathan numerator alone:
#
#     E[tokens/round](ρ, γ) = (1 - ρ^(γ+1)) / (1 - ρ)
# ════════════════════════════════════════════════════════════════════════════


def expected_speedup(rho: float, gamma: int) -> float:
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1]")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if rho >= 1.0 - 1e-12:
        return float(gamma + 1)
    num = 1.0 - rho ** (gamma + 1)
    den = (1.0 - rho) * (1.0 + gamma * (1.0 - rho))
    return float(num / den)


def expected_tokens_per_step(rho: float, gamma: int) -> float:
    if rho >= 1.0 - 1e-12:
        return float(gamma + 1)
    return float((1.0 - rho ** (gamma + 1)) / (1.0 - rho))


# ════════════════════════════════════════════════════════════════════════════
# Real AutoProfile invocation. We synthesize a model whose vocab + layered-ness
# matches the request, then call the actual library function.
# ════════════════════════════════════════════════════════════════════════════


def _build_model(name: str, vocab_size: int, layered: bool):
    """Construct a model instance whose surface AutoProfile inspects."""
    if layered:
        # MockLayeredModel has a class-level VOCAB_SIZE; subclass per-call so
        # AutoProfile sees the requested vocab without mutating the canonical.
        class _Layered(MockLayeredModel):
            @property
            def vocab_size(self_inner) -> int:  # noqa: N805
                return int(vocab_size)

        return _Layered(num_layers=24)
    if vocab_size != MockModel.VOCAB_SIZE:

        class _Mock(MockModel):
            @property
            def vocab_size(self_inner) -> int:  # noqa: N805
                return int(vocab_size)

        return _Mock()
    return MockModel()


def real_recommend(name: str, vocab_size: int, has_draft: bool, layered: bool) -> dict:
    model = _build_model(name, vocab_size, layered)
    profile = AutoProfile.recommend(model, name_hint=name, has_draft=has_draft)
    return {
        "strategy": profile.strategy,
        "gamma": profile.gamma,
        "early_exit_threshold": profile.early_exit_threshold,
        "temperature": profile.temperature,
        "use_cache": profile.use_cache,
        "cache_capacity": profile.cache_capacity,
        "rationale": profile.rationale,
        "model_inspected": {
            "name_hint": name,
            "vocab_size": int(model.vocab_size),
            "layered": layered,
            "has_draft": has_draft,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# Real speculative-decoding simulation.
#
# Standard generation: 1 target call per token → n_tokens steps.
# Speculative generation (Leviathan / Chen et al. 2023):
#   for each round:
#     1. draft proposes γ tokens
#     2. target verifies them in parallel — each accepted iid with prob ρ
#        (Bernoulli sampling at the rejection-sampling boundary)
#     3. on the first reject, that round emits (k accepted) + 1 fallback,
#        and the round ends after k+1 tokens with k+1 target-call equivalents
#        consumed (1 batched verify + 0 bonus)
#     4. if all γ accepted, emit γ + 1 bonus → γ+1 tokens for 1 round
#
# We count rounds (= target verification batches) as the kairu_steps measure.
# Standard steps = n_tokens. Speedup = standard_steps / kairu_steps.
# ════════════════════════════════════════════════════════════════════════════


def real_simulate_race(rho: float, gamma: int, n_tokens: int, seed: int = 42) -> dict:
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1]")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")

    rng = np.random.default_rng(seed)
    scheduler = DynamicGammaScheduler(initial=gamma, max_gamma=max(gamma, 8))

    rounds: list[dict] = []
    tokens_emitted = 0
    kairu_rounds = 0
    total_accepted = 0
    total_attempted = 0

    while tokens_emitted < n_tokens:
        kairu_rounds += 1
        # We respect the scheduler — but pin its window so γ does not change
        # on every test run; exposing the live γ is useful in the response.
        g = min(scheduler.gamma, n_tokens - tokens_emitted)
        if g < 1:
            g = 1

        # γ Bernoulli draws determine accept / reject per draft token.
        accept_mask = (rng.random(g) < rho).tolist()
        first_reject = next((i for i, a in enumerate(accept_mask) if not a), None)
        if first_reject is None:
            accepted_count = g
            bonus = 1  # all accepted ⇒ +1 bonus token from target dist
        else:
            accepted_count = first_reject
            bonus = 1  # the rejected slot's residual sample also yields a token
            # Truncate the mask for clarity in the trace.
            accept_mask = accept_mask[: first_reject + 1]

        round_tokens = accepted_count + bonus
        if tokens_emitted + round_tokens > n_tokens:
            round_tokens = n_tokens - tokens_emitted

        tokens_emitted += round_tokens
        total_accepted += accepted_count
        total_attempted += g

        scheduler.update(min(accepted_count, g), g)

        rounds.append(
            {
                "round": kairu_rounds,
                "gamma_used": g,
                "draft_tokens_proposed": g,
                "accepted_mask": accept_mask,
                "accepted_count": accepted_count,
                "tokens_emitted_this_round": round_tokens,
                "running_total_tokens": tokens_emitted,
                "scheduler_gamma_after": scheduler.gamma,
            }
        )

    standard_steps = n_tokens
    actual_speedup = standard_steps / kairu_rounds if kairu_rounds else float("inf")
    empirical_acceptance = total_accepted / total_attempted if total_attempted else 0.0
    closed_form_speedup = expected_speedup(rho, gamma)

    return {
        "n_tokens": n_tokens,
        "rho_input": rho,
        "gamma_input": gamma,
        "standard_steps": standard_steps,
        "kairu_steps": kairu_rounds,
        "actual_speedup": round(actual_speedup, 4),
        "closed_form_speedup": round(closed_form_speedup, 4),
        "empirical_acceptance_rate": round(empirical_acceptance, 4),
        "scheduler_final_gamma": scheduler.gamma,
        "scheduler_adjustments": scheduler.stats()["adjustments"],
        "rounds": rounds,
        "seed": seed,
    }


# ════════════════════════════════════════════════════════════════════════════
# KV-cache simulation: exercise LogitsCache LRU operations.
# ════════════════════════════════════════════════════════════════════════════


def real_kv_cache_sim(capacity: int, n_ops: int, seed: int) -> dict:
    """Simulate LogitsCache LRU operations; return per-op trace + final stats."""
    if not (1 <= capacity <= 64):
        raise ValueError("capacity must be in [1, 64]")
    if not (1 <= n_ops <= 200):
        raise ValueError("n_ops must be in [1, 200]")
    cache = LogitsCache(capacity=capacity)
    rng = np.random.default_rng(seed)
    key_range = max(2, capacity + capacity // 2)
    dummy = rng.standard_normal(8).astype(np.float32)
    ops: list[dict] = []
    for i in range(n_ops):
        key_id = int(rng.integers(0, key_range))
        key = (key_id,)
        current_size = len(cache)
        hit = cache.get(key) is not None
        will_evict = (not hit) and (current_size >= capacity)
        if not hit:
            cache.put(key, dummy)
        ops.append(
            {
                "i": i,
                "key": key_id,
                "hit": hit,
                "eviction": will_evict,
                "size": len(cache),
            }
        )
    return {"ops": ops, "stats": cache.stats(), "capacity": capacity, "n_ops": n_ops}


# ════════════════════════════════════════════════════════════════════════════
# Early-exit simulation: run LayerwiseEarlyExitDecoder and return per-token
# exit-layer trace.
# ════════════════════════════════════════════════════════════════════════════


def real_early_exit_sim(
    threshold: float, n_tokens: int, num_layers: int, seed: int
) -> dict:
    """Run LayerwiseEarlyExitDecoder and return per-token exit-layer trace."""
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in (0, 1]")
    if not (1 <= n_tokens <= 100):
        raise ValueError("n_tokens must be in [1, 100]")
    if num_layers not in (6, 12, 24, 48):
        raise ValueError("num_layers must be one of 6, 12, 24, 48")

    class _SeedableLayeredModel(MockLayeredModel):
        def __init__(self, num_layers: int, seed: int) -> None:
            super().__init__(num_layers=num_layers)
            self._seed = seed

        def _base_logits(self, token_ids: list) -> np.ndarray:
            rng_seed = (sum(token_ids) * 2654435761 + self._seed) % (2**32)
            rng = np.random.default_rng(rng_seed)
            logits = rng.standard_normal(self.VOCAB_SIZE).astype(np.float32)
            preferred = (sum(token_ids) * 7 + 13) % self.VOCAB_SIZE if token_ids else 0
            logits[preferred] += 3.0
            return logits

    model = _SeedableLayeredModel(num_layers=num_layers, seed=seed)
    decoder = LayerwiseEarlyExitDecoder(
        model=model, confidence_threshold=threshold, min_layer=1
    )
    prompt = [1, 2, 3, 4, 5]
    tokens, stats = decoder.generate(prompt, max_new_tokens=n_tokens)
    exit_layers = stats["exit_layers"]
    per_token = [
        {
            "idx": i,
            "token_id": int(tok),
            "exit_layer": int(el),
            "depth_pct": round(el / num_layers, 4),
            "layers_saved": num_layers - int(el),
        }
        for i, (tok, el) in enumerate(zip(tokens, exit_layers))
    ]
    return {
        "tokens": per_token,
        "num_layers": num_layers,
        "n_tokens": len(tokens),
        "mean_exit_layer": round(stats["mean_exit_layer"], 3),
        "compute_saved": round(stats["compute_saved"], 4),
        "threshold": threshold,
    }


# ════════════════════════════════════════════════════════════════════════════
# Watermark demo: generate watermarked tokens then detect the watermark.
# ════════════════════════════════════════════════════════════════════════════


def real_watermark_demo(n_tokens: int, seed: int, delta: float, scheme: str) -> dict:
    """Generate watermarked tokens then detect the watermark.

    Returns per-token green-list membership plus detection statistics.
    """
    if not (1 <= n_tokens <= 100):
        raise ValueError("n_tokens must be in [1, 100]")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if scheme not in ("single", "context"):
        raise ValueError("scheme must be 'single' or 'context'")

    model = MockModel()
    processor = WatermarkLogitsProcessor(
        vocab_size=model.vocab_size, delta=delta, seeding_scheme=scheme
    )
    # StreamingDecoder always seeds at 42; use seed to shift the prompt
    prompt = [(seed % 97) + 1, (seed % 53) + 2, (seed % 31) + 3]
    decoder = StreamingDecoder(model, temperature=1.0, watermark=processor)
    tokens = decoder.generate(prompt, max_new_tokens=n_tokens)

    # Reconstruct green-list membership per token
    context = list(prompt)
    per_token = []
    for tok_id in tokens:
        gl = processor.green_list(context)
        per_token.append({"id": int(tok_id), "green": bool(gl[tok_id])})
        context.append(tok_id)

    # Run detector
    detector = WatermarkDetector(vocab_size=model.vocab_size, seeding_scheme=scheme)
    result = detector.detect(tokens, prefix_ids=prompt)

    return {
        "tokens": per_token,
        "n_tokens": len(tokens),
        "z_score": round(result.z_score, 4),
        "p_value": round(result.p_value, 6),
        "decision": result.decision,
        "green_fraction": round(result.green_fraction, 4),
        "num_green": result.num_green,
        "threshold": result.threshold,
        "delta": delta,
        "scheme": scheme,
    }


# ════════════════════════════════════════════════════════════════════════════
# Prism: run all eight named rubrics on one (prompt, response) and return
# the ordered list of beam payloads the UI renders.  Each entry carries the
# rubric name, its canonical color, its aggregate score in [0, 1], and the
# per-criterion sub-scores so hover tooltips can explain the result.
# ════════════════════════════════════════════════════════════════════════════


def _prism_beams(prompt: str, response: str) -> list:
    out = []
    for name in rubric_names():
        ev = eval_one(prompt, response, rubric=name)
        out.append(
            {
                "rubric": name,
                "color": RUBRIC_DEFS[name]["color"],
                "score": round(ev.aggregate, 4),
                "description": RUBRIC_DEFS[name]["description"],
                "components": {s.name: round(s.score, 4) for s in ev.scores},
            }
        )
    return out


# ════════════════════════════════════════════════════════════════════════════
# HTTP plumbing — stdlib only.
# ════════════════════════════════════════════════════════════════════════════


class DemoHandler(BaseHTTPRequestHandler):
    server_version = f"kairu-demo/{KAIRU_VERSION}"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than default
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e

    # ─── routing ────────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:  # noqa: N802 — required signature
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            return self._serve_index()
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "version": KAIRU_VERSION})
        if self.path == "/api/rubrics":
            return self._json(
                200,
                {
                    "rubrics": [
                        {
                            "name": n,
                            "description": RUBRIC_DEFS[n]["description"],
                            "color": RUBRIC_DEFS[n]["color"],
                            "weights": dict(RUBRIC_DEFS[n]["weights"]),
                        }
                        for n in rubric_names()
                    ],
                },
            )
        return self._json(404, {"error": "not found", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
        except ValueError as e:
            return self._json(400, {"error": str(e)})

        try:
            if self.path == "/api/speedup":
                return self._handle_speedup(body)
            if self.path == "/api/recommend":
                return self._handle_recommend(body)
            if self.path == "/api/simulate-race":
                return self._handle_simulate(body)
            if self.path == "/api/prism":
                return self._handle_prism(body)
            if self.path == "/api/kv-cache-sim":
                return self._handle_kv_cache(body)
            if self.path == "/api/early-exit-sim":
                return self._handle_early_exit(body)
            if self.path == "/api/watermark-demo":
                return self._handle_watermark(body)
        except (ValueError, TypeError, KeyError) as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.exception("internal error")
            return self._json(500, {"error": f"internal: {e}"})

        return self._json(404, {"error": "not found", "path": self.path})

    # ─── handlers ───────────────────────────────────────────────────────
    def _serve_index(self) -> None:
        if not INDEX_PATH.exists():
            return self._json(500, {"error": f"missing {INDEX_PATH}"})
        body = INDEX_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _handle_speedup(self, body: dict) -> None:
        rho = float(body.get("rho", 0.7))
        gamma = int(body.get("gamma", 4))
        s = expected_speedup(rho, gamma)
        tps = expected_tokens_per_step(rho, gamma)
        formula = "(1 - ρ^(γ+1)) / ((1 - ρ) · (1 + γ · (1 - ρ)))"
        derivation = (
            f"numerator   = 1 - ρ^(γ+1)    = 1 - {rho}^({gamma}+1) = {1 - rho ** (gamma + 1):.6f}"
            f" | denom1 = 1 - ρ = {1 - rho:.6f}"
            f" | denom2 = 1 + γ·(1-ρ) = {1 + gamma * (1 - rho):.6f}"
            f" | speedup = {s:.4f}"
        )
        return self._json(
            200,
            {
                "speedup": round(s, 6),
                "expected_tokens_per_step": round(tps, 6),
                "formula_used": formula,
                "derivation": derivation,
                "inputs": {"rho": rho, "gamma": gamma},
                "source": "kairu.gamma_scheduler.DynamicGammaScheduler (Leviathan et al. 2023)",
            },
        )

    def _handle_recommend(self, body: dict) -> None:
        name = str(body.get("model_name", body.get("name", "")))
        vocab_size = int(body.get("vocab_size", body.get("context_len", 32_000)))
        has_draft = bool(body.get("has_draft", False))
        layered = bool(body.get("layered", False))
        result = real_recommend(
            name=name, vocab_size=vocab_size, has_draft=has_draft, layered=layered
        )
        result["source"] = "kairu.auto_profile.AutoProfile.recommend (live call)"
        return self._json(200, result)

    def _handle_prism(self, body: dict) -> None:
        prompt = body.get("prompt")
        response = body.get("response")
        response_b = body.get("response_b")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("prompt and response must be strings")
        if not prompt or not response:
            raise ValueError("prompt and response must be non-empty")
        if len(prompt) > PRISM_MAX_TEXT or len(response) > PRISM_MAX_TEXT:
            raise ValueError(f"text exceeds {PRISM_MAX_TEXT} chars")
        if response_b is not None:
            if not isinstance(response_b, str) or not response_b:
                raise ValueError("response_b must be a non-empty string when provided")
            if len(response_b) > PRISM_MAX_TEXT:
                raise ValueError(f"response_b exceeds {PRISM_MAX_TEXT} chars")
        beams_a = _prism_beams(prompt, response)
        out: dict = {"beams_a": beams_a, "rubric_order": list(rubric_names())}
        if response_b is not None:
            beams_b = _prism_beams(prompt, response_b)
            out["beams_b"] = beams_b
            agg_a = sum(b["score"] for b in beams_a) / len(beams_a)
            agg_b = sum(b["score"] for b in beams_b) / len(beams_b)
            margin = abs(agg_a - agg_b)
            winner = (
                "a"
                if agg_a - agg_b > 0.005
                else ("b" if agg_b - agg_a > 0.005 else "tie")
            )
            out["aggregate_a"] = round(agg_a, 4)
            out["aggregate_b"] = round(agg_b, 4)
            out["margin"] = round(margin, 4)
            out["winner"] = winner
        return self._json(200, out)

    def _handle_simulate(self, body: dict) -> None:
        rho = float(body.get("rho", 0.8))
        gamma = int(body.get("gamma", 4))
        n_tokens = int(body.get("n_tokens", 50))
        seed = int(body.get("seed", 42))
        if n_tokens > 500:
            raise ValueError("n_tokens capped at 500 for the demo")
        result = real_simulate_race(rho=rho, gamma=gamma, n_tokens=n_tokens, seed=seed)
        result["source"] = (
            "kairu.gamma_scheduler.DynamicGammaScheduler + numpy seeded RNG; "
            "rejection-sampling per Leviathan et al. 2023 §3"
        )
        return self._json(200, result)

    def _handle_kv_cache(self, body: dict) -> None:
        capacity = int(body.get("capacity", 8))
        n_ops = int(body.get("n_ops", 50))
        seed = int(body.get("seed", 42))
        result = real_kv_cache_sim(capacity=capacity, n_ops=n_ops, seed=seed)
        result["source"] = "kairu.kv_cache.LogitsCache (live simulation)"
        return self._json(200, result)

    def _handle_early_exit(self, body: dict) -> None:
        threshold = float(body.get("threshold", 0.85))
        n_tokens = int(body.get("n_tokens", 30))
        num_layers = int(body.get("num_layers", 12))
        seed = int(body.get("seed", 42))
        result = real_early_exit_sim(
            threshold=threshold, n_tokens=n_tokens, num_layers=num_layers, seed=seed
        )
        result["source"] = "kairu.layered.LayerwiseEarlyExitDecoder (live simulation)"
        return self._json(200, result)

    def _handle_watermark(self, body: dict) -> None:
        n_tokens = int(body.get("n_tokens", 40))
        seed = int(body.get("seed", 42))
        delta = float(body.get("delta", 2.0))
        scheme = str(body.get("scheme", "single"))
        result = real_watermark_demo(
            n_tokens=n_tokens, seed=seed, delta=delta, scheme=scheme
        )
        result["source"] = (
            "kairu.watermark.WatermarkLogitsProcessor + WatermarkDetector "
            "(Kirchenbauer et al. 2023)"
        )
        return self._json(200, result)


# ════════════════════════════════════════════════════════════════════════════
# Cache demo — exercise LogitsCache during startup so the banner has real data
# ════════════════════════════════════════════════════════════════════════════


def _selftest_cache() -> dict:
    cache = LogitsCache(capacity=4)
    rng = np.random.default_rng(0)
    for k in range(6):
        cache.put((k,), rng.standard_normal(8).astype(np.float32))
    cache.get((4,))
    cache.get((5,))
    cache.get((0,))  # 2 hits, 1 miss
    return cache.stats()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kairu interactive demo server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cache_stats = _selftest_cache()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print()
    print("  \033[1;36m流  K A I R U   D E M O\033[0m")
    print(f"  kairu v{KAIRU_VERSION}")
    print(f"  serving demo at  \033[1;32mhttp://{args.host}:{args.port}/\033[0m")
    print()
    print("  endpoints (all JSON unless noted):")
    print("    GET   /                  → demo/index.html")
    print("    GET   /api/health        → liveness probe")
    print("    POST  /api/speedup       → real speedup formula")
    print("    POST  /api/recommend     → AutoProfile.recommend()")
    print("    POST  /api/simulate-race → seeded speculative-decoding sim")
    print("    GET   /api/rubrics       → list 8 named prism rubrics")
    print("    POST  /api/prism         → all 8 rubric scores for a (prompt, response)")
    print("    POST  /api/kv-cache-sim      → LogitsCache simulation")
    print("    POST  /api/early-exit-sim    → LayerwiseEarlyExitDecoder simulation")
    print("    POST  /api/watermark-demo    → watermark apply + detect")
    print()
    print(f"  startup self-test  LogitsCache(4)  →  {cache_stats}")
    print("  Ctrl-C to stop")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  \033[2mshutting down\033[0m")
        server.shutdown()


if __name__ == "__main__":
    main()
