# Kairu — Project Roadmap

> 流 · *to flow, to stream*

Current version: **v0.6.0**

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
- Boundary validation: empty/oversized prompts, control characters, max-tokens cap, temperature ∈ [0, 2], non-negative `stop_token_id`
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

## Phase 7 — Observability & Real-Workload Validation (v0.7.0)

- OpenTelemetry trace export from `/generate` (per-token spans, propagating client trace context)
- Token-budget enforcement at the cluster scope via shared Redis counters
- Real benchmark harness against `tiny-gpt2` + `gpt2` on a fixed 100-prompt corpus, scripted to publish a results JSON to `benchmarks/results/`
- Helm chart + `kustomize` manifests for the GHCR image
- Streaming `/generate` JSONL fallback for clients that cannot consume SSE
