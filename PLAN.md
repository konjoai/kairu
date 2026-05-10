# Kairu — Project Roadmap

> 流 · *to flow, to stream*

Current version: **v0.12.0**

---

## Phase 1 — Core Engine (v0.1.0) ✅ COMPLETE

**Ship Gate:** 31 Python tests passing.

Deliverables:
- `ModelInterface` abstract base — zero-dependency contract for any model backend
- `MockModel` — deterministic LCG-seeded mock; enables full test coverage without ML frameworks
- `SpeculativeDecoder` — draft-model lookahead with acceptance-ratio rejection sampling (Chen et al. 2023)
- `EarlyExitDecoder` — confidence-threshold + entropy-floor halting
- `TokenBudget` — hard prompt+generation cap with `consume()`, `remaining`, `utilization()`
- `GenerationMetrics` — wall-clock timing, tok/s, mean latency, acceptance rate
- `KairuDashboard` — Rich live panel for real-time metric display
- `ModelWrapper` + `wrap_model()` — unified entry point wiring all layers together
- `HuggingFaceModel` / `_hf_backend.py` — optional HF integration (behind `kairu[hf]`)
- `pyproject.toml` with `hatchling` build, `dev` + `hf` extras
- GitHub Actions CI — Python 3.11, `pytest -v`

---

## Phase 2 — HuggingFace Integration (v0.2.0) ✅ COMPLETE

**Ship Gate:** 51 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/tokenizer.py` — `TokenizerBase` ABC, `MockTokenizer` (deterministic, no deps), `HFTokenizer` (wraps HF `AutoTokenizer`)
- `kairu/streaming.py` — `StreamingDecoder`: greedy/temperature-sampled token-by-token iterator using only NumPy + `ModelInterface`; `stream()` yields IDs, `generate()` collects to list; stop-token support
- `kairu/_hf_backend.py` — full rewrite: `HuggingFaceModel` with `encode()`, `decode()`, `stream_generate()` (HF `TextIteratorStreamer` + threading); all heavy imports deferred to `__init__` so the module is importable without ML libs
- `kairu/__init__.py` — exports `StreamingDecoder`, `MockTokenizer`, `TokenizerBase`; guarded import of `HFTokenizer`; version bumped to `0.2.0`
- 8 tokenizer tests (`tests/test_tokenizer.py`) — fully offline
- 8 streaming tests (`tests/test_streaming.py`) — uses `MockModel`, no HF
- 8 HF backend tests (`tests/test_hf_backend.py`) — 4 structural (offline), 4 integration gated behind `KAIRU_TEST_HF=1`

---

## Phase 3 — Benchmarking (v0.3.0) ✅ COMPLETE

**Ship Gate:** 59 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/bench.py` — `BenchmarkRunner` driving N generation runs against any `ModelInterface`; `build_parser()` + `main()` CLI entry; p50/p95/p99/stddev via pure `statistics` + sorted-list percentile (no scipy)
- `kairu/bench.BenchmarkResult` — dataclass with `latencies_s`, `p50`, `p95`, `p99`, `mean`, `stddev`, `tokens_per_s_mean`, `hardware`, `timestamp`; `to_json()` + `save()` (never overwrites)
- `kairu/bench._collect_hardware()` — hostname, OS, machine, CPU model (`sysctl`/`/proc/cpuinfo`), total RAM (`psutil`/`sysctl hw.memsize`/`/proc/meminfo`), Python version
- `kairu/__main__bench.py` — thin re-export shim for `build_parser`, `main`, `_collect_hardware`
- `python -m kairu.bench --model mock --tokens 100 --runs 50 --warmup 5` exits 0 with no ML deps
- 8 benchmark tests in `tests/test_bench.py` — shape, percentile ordering, JSON round-trip, file save, hardware keys, CLI exit 0, CLI --help, filename contains timestamp+name
- `kairu/__init__.py` exports `BenchmarkRunner`, `BenchmarkResult`; version bumped `0.2.0 → 0.3.0`

---

## Phase 4 — Streaming API (v0.4.0) ✅ COMPLETE

**Ship Gate:** 73 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/server.py` — `create_app(model?, tokenizer?, config?)` FastAPI factory; `POST /generate` SSE endpoint, `GET /health`; OpenAI-compatible `chat.completion.chunk` frames + `kairu` extension carrying per-token `token_id`/`index`/`latency_ms`/`tokens_per_s`; final frame's `finish_reason ∈ {length, stop, timeout}`; trailing `data: [DONE]\n\n` sentinel
- `kairu.server.ServerConfig` — `max_prompt_chars`, `max_tokens_cap`, `request_timeout_s`, `rate_limit_requests`, `rate_limit_window_s`; every limit enforced at the API boundary before the tokenizer is touched
- `kairu.server.RateLimiter` — pure-stdlib sliding-window per-key limiter, `asyncio.Lock`-guarded
- Boundary validation: empty/oversized prompts, control characters, max-tokens cap, temperature ∈ [0, 2], non-positive `stop_token_id`
- SHA-256-only prompt logging (raw content never logged)
- 14 server tests in `tests/test_server.py` — health, OpenAI chunk shape, `[DONE]` sentinel, all validation paths, 429 rate limiting, request timeout, sliding-window unit tests, total_s monotonicity
- `pyproject.toml` — new `server` extras (`fastapi`, `uvicorn`, `pydantic`); `dev` extras gain `pytest-asyncio` and `httpx`; `asyncio_mode = "auto"`
- `kairu/__init__.py` — guarded re-export of `create_app`, `ServerConfig`, `RateLimiter`; version `0.3.0 → 0.4.0`

---

## Phase 5 — Model-Aware Optimization (v0.5.0) ✅ COMPLETE

**Ship Gate:** 112 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/layered.py` — `LayeredModelInterface` extension; `MockLayeredModel` with depth-monotonic confidence; `LayerwiseEarlyExitDecoder` reporting per-token exit layers and `compute_saved`
- `kairu/kv_cache.py` — `LogitsCache` (bounded LRU, O(1) get/put, hit/miss/evict stats); `CachedModel` wrapper memoizing `next_token_logits` keyed by prefix tuple
- `kairu/gamma_scheduler.py` — `DynamicGammaScheduler` (AIMD over γ, configurable bounds/thresholds/window/rates)
- `kairu/auto_profile.py` — `AutoProfile.recommend(model, name_hint?, has_draft=False)` → frozen `DecoderProfile{strategy, gamma, threshold, temperature, use_cache, cache_capacity, rationale}`
- `kairu/speculative.py` — optional `scheduler` kwarg; per-round `scheduler.update`; stats now include `final_gamma` and `gamma_adjustments`
- `kairu/wrapper.py` — new flags `cache_capacity` (transparently wraps target+draft in `CachedModel`) and `adaptive_gamma` (auto-attaches scheduler)
- 39 new tests across `tests/test_layered.py`, `tests/test_kv_cache.py`, `tests/test_gamma_scheduler.py`, `tests/test_auto_profile.py`
- `kairu/__init__.py` — exports new types; version `0.4.0 → 0.5.0`

---

## Phase 6 — Distributed & Production Hardening (v0.6.0) ✅ COMPLETE

**Ship Gate:** 139 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/rate_limit.py` — `RateLimiterBackend` protocol; `InMemoryBackend` (default, BC-preserving) + `RedisBackend` (atomic `MULTI` pipeline with speculative-add rollback). `kairu.create_app(..., rate_limit_backend=...)` accepts any backend.
- `kairu/metrics_export.py` — pure-stdlib Prometheus exposition; `Counter`/`Gauge`/`Histogram` (canonical Prometheus latency buckets, O(log buckets) `observe`). `MetricsCollector` exposes the named series the dashboard contract depends on.
- `GET /metrics` endpoint on the SSE server, instrumenting `/health`, `/generate` success/422/429/500, and the active-streams gauge with proper `try/finally`.
- `kairu/cli.py` — `kairu serve | bench | version` console script. `serve` covers every `ServerConfig` field plus `--cache-capacity`, `--adaptive-gamma`, `--redis URL`.
- `kairu/_hf_backend.py` — new `HuggingFaceKVCachedModel`; persistent `past_key_values` keyed by longest-common-prefix between successive calls; `kv_cache_stats` and `reset_cache()` exposed.
- `Dockerfile` (multi-stage slim, non-root uid 1001, healthcheck) + `.dockerignore`.
- `.github/workflows/docker.yml` — multi-arch (amd64/arm64) GHCR publish on `main` push and `v*.*.*` tags. Buildx + QEMU + GHA cache. Forks safe.
- 27 new tests across `tests/test_rate_limit.py` (12), `tests/test_metrics_export.py` (8), `tests/test_cli.py` (7); plus 2 server-side `/metrics` tests.
- `kairu/__init__.py` — exports `MetricsCollector`, `InMemoryBackend`, `RedisBackend`, `RateLimiterBackend`; version `0.5.0 → 0.6.0`. `pyproject.toml` adds `redis` extra and `kairu` console script.

---

## Phase 7 — Observability & Real-Workload Validation (v0.7.0) ✅ COMPLETE

**Ship Gate:** 191 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/tracing.py` — `KairuTracer` (OTel API facade + automatic `_NoOpTracer` fallback when `opentelemetry-api` is absent). W3C `traceparent`/`tracestate` extraction via `extract_trace_context(headers)`. Per-token `add_event` annotations on the `kairu.generate` root span (not child spans — avoids trace store bloat). `record_generation_complete()` / `record_error()` helpers. `headers_from_request()` normalises ASGI headers for propagation.
- `kairu/cluster_budget.py` — `ClusterTokenBudget` (Redis `INCRBY`/`DECRBY`/`EXPIRE` with speculative-rollback on cap overflow) + `LocalClusterBudget` (in-process, `asyncio.Lock`-guarded, window-resetting). Both implement the `ClusterBudgetBackend` protocol. Configurable scope string for multi-tenant isolation.
- `kairu/server.py` — JSONL streaming fallback: when the client sets `Accept: application/x-ndjson` the server emits the same frame objects as newline-delimited JSON (no `data:` prefix, no `[DONE]` sentinel). OTel tracing wired into the `_token_loop` shared generator — SSE and JSONL paths both get per-token span events. `create_app()` now accepts an optional `tracer: KairuTracer` kwarg. Server `version` bumped to `0.7.0`.
- `benchmarks/run_corpus.py` — `CorpusBenchmarkRunner` driving the full 100-prompt corpus (instruction-following, Q&A, coding, summarisation, free-form) against any `ModelInterface`. `--model mock` runs fully offline. Results saved via `BenchmarkResult.save()` to `benchmarks/results/` (never overwrites). Standalone CLI: `python benchmarks/run_corpus.py --model mock --tokens 64 --runs 100`.
- `helm/kairu/` — Helm chart v0.7.0: `Chart.yaml`, `values.yaml` (image, replicas, resources, probes, autoscaling, Redis, OTel, ServiceMonitor, PDB toggles), templates (`deployment.yaml`, `service.yaml`, `configmap.yaml`, `hpa.yaml`, `ingress.yaml`, `servicemonitor.yaml`, `pdb.yaml`, `_helpers.tpl`).
- `kustomize/` — Kustomize base (Deployment + Service) + overlays: `production` (4 replicas, doubled resources) and `staging` (1 replica).
- 52 new tests: `tests/test_tracing.py` (13), `tests/test_cluster_budget.py` (19), `tests/test_jsonl_stream.py` (10), `tests/test_corpus_bench.py` (10). All run offline with `MockModel`.
- `kairu/__init__.py` — exports `KairuTracer`, `extract_trace_context`, `ClusterTokenBudget`, `LocalClusterBudget`; version `0.6.0 → 0.7.0`.
- `pyproject.toml` — version `0.6.0 → 0.7.0`; new `otel` optional extra (`opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`).

---

## Phase 8 — Adaptive Router (v0.8.0) ✅ COMPLETE

**Ship Gate:** 221 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/router.py` — `DecoderRouter`: given prompt token IDs and runtime signals (prompt length, draft-model availability, latency budget), deterministically selects the optimal decoding strategy (`streaming` / `speculative` / `early_exit`). Routing rules: tight budget (< 200 ms) → streaming; short prompt (< threshold) → streaming; draft available → speculative; otherwise → early_exit. EWMA latency tracking per strategy via `record_outcome(decision, metrics)`.
- `kairu/feedback.py` — `FeedbackLoop`: ingests `BenchmarkResult` objects, buffers until `min_results` reached, then computes mean acceptance rate and drives `DynamicGammaScheduler` up (high AR > 0.75) or down (low AR < 0.40) via 100 %/0 % synthetic update rounds. Emits `FeedbackSummary` on each flush cycle.
- `RouterDecision` dataclass — carries `strategy`, `profile` (a `DecoderProfile` with strategy overridden to match routing), `confidence`, `rationale`, `latency_budget_ms`.
- `RoutingStats` dataclass — per-strategy decision counts, per-strategy EWMA latency, total routed.
- `FeedbackSummary` dataclass — `n_results`, `mean_acceptance_rate`, `gamma_adjusted`, `new_gamma`, `recommendation`.
- 16 router tests in `tests/test_router.py` — construction, all routing branches, budget override, stats accumulation, EWMA update, profile strategy alignment.
- 14 feedback tests in `tests/test_feedback.py` — flush threshold, buffer clearing, gamma direction, summary fields, multi-cycle operation.
- `kairu/__init__.py` — exports `DecoderRouter`, `RouterDecision`, `RoutingStats`, `FeedbackLoop`, `FeedbackSummary`; version `0.7.0 → 0.8.0`.

---

## Phase 9 — Token Watermarking & Integrity (v0.9.0) ✅ COMPLETE

**Ship Gate:** 239 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/watermark.py` — Kirchenbauer et al. (2023) green/red list watermarking scheme implemented with pure NumPy + stdlib (`hashlib`, `math`, `struct`). No ML framework dependency.
  - `WatermarkLogitsProcessor` — at each decoding step, hash-seeds a green/red partition of the vocabulary using the preceding token (or a context window) then adds a scalar bias `δ` to all green-list logits before softmax. Supports `seeding_scheme ∈ {"single", "context"}`. Never mutates the input logits array.
  - `WatermarkDetector` — given a token sequence + optional prompt prefix, reconstructs the per-step green lists (identical seeding parameters) and counts green tokens. Computes z-score against the Binomial(T, 0.5) null (well-approximated by N for T ≥ 20) and one-sided p-value via `math.erfc` (no scipy). Returns a frozen `WatermarkResult` dataclass.
  - `WatermarkResult` — frozen dataclass carrying `num_tokens`, `num_green`, `green_fraction`, `z_score`, `p_value`, `decision`, `threshold`.
  - `_norm_sf(z)` — exact normal survival function via `math.erfc`; no scipy dependency.
- `kairu/streaming.py` — `StreamingDecoder` gains optional `watermark: WatermarkLogitsProcessor | None` constructor kwarg. When set, `process()` is called on every logit array before sampling; when None the code path is identical to v0.8.0 (zero overhead).
- 18 new tests in `tests/test_watermark.py` — covering: seed determinism, seed uniqueness, green-list shape/fraction/reproducibility, processor construction validation, logit bias correctness, immutability of input, shape-mismatch error, empty context, scheme divergence, detector construction, empty sequence error, result fields, watermarked-sequence z-score direction, unwatermarked no false positive, frozen result mutation, `_norm_sf` edge cases and monotonicity.
- `kairu/__init__.py` — exports `WatermarkLogitsProcessor`, `WatermarkDetector`, `WatermarkResult`; version `0.8.0 → 0.9.0`.
- `pyproject.toml` — version `0.8.0 → 0.9.0`; description updated.

---

## Phase 11 — Squish Integration: Quantization-Tier Quality Eval (v0.10.0) ✅ COMPLETE

**Ship Gate:** 286 Python tests passing (4 gated HF integration tests skipped without `KAIRU_TEST_HF=1`).

Deliverables:
- `kairu/squish_eval.py` — pure-stdlib quality rubric for evaluating LLM outputs across quantization tiers (FP16 baseline vs INT8/INT4/INT2 quantized).
  - 5-criterion `SquishEvaluator` (correctness, fluency, faithfulness, completeness, safety) — every score in `[0, 1]`, aggregate is unweighted mean.
  - `quality_degradation_report(...)` — per-criterion delta + retention% per tier; tiers ordered by descending bit-width.
  - `recommended_quant_tier(report, tolerance)` — returns the deepest tier still within `(1 - tolerance) * 100`% retention.
  - All result types are frozen dataclasses with `as_dict()` for JSON transport.
- `POST /compare/quantization` in `kairu/server.py` — Pydantic-validated, rate-limited, pure-CPU endpoint that wraps the module-level functions and returns `{report, recommended_tier, tolerance}`.
- `demo/sample_comparisons/04_quantization_comparison.json` — realistic GPT-4 vs INT8/INT4/INT2 sample over 5 prompts with verified retention numbers and ready-to-POST request body.
- 20 new tests (`tests/test_squish_eval.py` + `tests/test_server.py`) covering identical/degraded/garbage tiers, empty inputs, length mismatches, recommendation thresholds, frozen-dataclass invariants, JSON round-trip, and HTTP-level integration.
- `kairu/__init__.py` and `pyproject.toml`: version `0.9.0 → 0.10.0`; description updated.

---

## Phase 12 — Evaluation API & A/B Comparison (v0.11.0) ✅ COMPLETE

**Ship Gate:** 294 Python tests passing, 4 skipped HF-gated (32 evaluation + 16 HTTP-boundary tests added).

Deliverables:
- `kairu/evaluation.py` — seven heuristic scorers (relevance F1, coherence trigram-uniqueness, conciseness Gaussian, safety regex categories, fluency sentence-length + TTR, specificity entity density, completeness recall). All deterministic, pure-stdlib, bounded to [0, 1].
- Five built-in rubrics: `default`, `helpfulness`, `safety_focused`, `concise_qa`, `creative` — composable with `weights={…}` overrides per call.
- `compare()` returns `Comparison` with absolute margin and per-criterion winner using a `TIE_EPSILON = 0.005` noise floor.
- `evaluate_batch()` + `to_csv()` — batch driver returning JSON or CSV-ready rows.
- `api/main.py` — FastAPI app exposing `POST /evaluate`, `POST /compare`, `GET /rubrics`, `POST /batch`, `GET /health`; pydantic v2 models, 413 on oversize, 422 on bad rubric/criterion.
- `api/Dockerfile` (slim, non-root, `$PORT`-aware), `api/requirements.txt`, `render.yaml` — deployable to Render / Fly / GKE.
- `demo/sample_comparisons/` — 3 runnable A/B fixtures with `expected_winner` for regression validation.
- `kairu/__init__.py` — re-exports `evaluate`, `compare`, `evaluate_batch`, `to_csv`, `Evaluation`, `Comparison`, `Rubric`, `CRITERIA`, `RUBRICS`; version `0.10.0 → 0.11.0`.

---

## Phase 13 — Eight Named Rubrics + Prism UI (v0.12.0) ✅ COMPLETE

**Ship Gate:** 313 tests passing, 4 HF-gated skipped (13 rubric + 6 API-route tests added).

Deliverables:
- `kairu/rubrics.py` — `RUBRIC_DEFS`: eight named rubrics (helpfulness, accuracy, safety, coherence, conciseness, creativity, groundedness, tone) with curated weights + canonical hex color per rubric.
- `kairu/evaluation.RUBRICS` auto-materialises from `RUBRIC_DEFS`.
- API: `GET /rubrics/{name}` + `POST /evaluate/rubric/{name}` (path-param routing).
- `demo/server.py` — `POST /api/prism` runs all eight rubrics on one (prompt, response[, response_b]); pure stdlib, 16 KB input cap.
- `demo/index.html` — full rebuild as the prism UI: pure dark `#06060f`, idly-rotating SVG triangular prism, eight color-coded beams, A/B mode with offset dashed beams, click-to-evaluate, hover tooltips, kbd shortcut.
- 19 new tests across `tests/test_rubrics.py` and `api/test_api.py`.

---

## Phase 14 — Prompt Shield & Content Policy 🔜 NEXT

**Goal:** Production-safe content screening at the API boundary, prior to tokenization.

Proposed deliverables:
- `kairu/shield.py` — `PromptShield`: rule-based (regex + keyword) + configurable heuristic classifier. Checks for prompt-injection patterns, PII leakage templates, jailbreak canonical phrases. Returns a `ShieldVerdict` (allowed / blocked / flagged) with a `reason` string and `confidence` float.
- `ShieldConfig` dataclass — configurable rule sets, PII patterns, injection signatures, custom allow/block lists.
- Server integration: `create_app()` accepts optional `shield: PromptShield`; if present it runs *before* rate limiting (fail-fast, cheapest guard first).
- 16+ tests in `tests/test_shield.py` — all offline, covering clean prompts, injection patterns, PII patterns, custom rules, server integration.
