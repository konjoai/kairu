# Changelog

All notable changes to Kairu follow [Conventional Commits](https://www.conventionalcommits.org/) and [Keep a Changelog](https://keepachangelog.com/) conventions.

---

## [0.6.0] — 2026-05-03

### Added — Distributed & Production Hardening

- `kairu.rate_limit.RateLimiterBackend` — pluggable storage protocol for sliding-window rate limit state. Two backends ship: `InMemoryBackend` (single-process, default, identical behavior to v0.4.0) and `RedisBackend` (cross-process via Redis sorted sets with atomic `MULTI`/`ZREMRANGEBYSCORE`/`ZCARD`/`ZADD` pipeline; rolls back the speculative add when the count is over). Public `RateLimiter` API unchanged — `kairu.create_app(..., rate_limit_backend=...)` accepts any backend.
- `kairu.metrics_export` — pure-stdlib Prometheus exposition format. Three primitives (`Counter`, `Gauge`, `Histogram`) with thread-safe mutations; `Histogram` uses canonical Prometheus latency buckets (0.005 s … 10 s) with O(log buckets) `observe`. `MetricsCollector` holds the named series the dashboard contract depends on: `kairu_requests_total`, `kairu_tokens_generated_total`, `kairu_errors_total`, `kairu_rate_limited_total`, `kairu_active_streams`, `kairu_request_duration_seconds`, `kairu_token_latency_seconds`, `kairu_process_uptime_seconds`.
- `GET /metrics` endpoint — emits the exposition payload with `Content-Type: text/plain; version=0.0.4; charset=utf-8`. Wired into `/health`, `/generate` (success, validation, rate-limit, stream failure), and the active-streams gauge with proper `try/finally` on the SSE generator.
- `kairu.cli` — new console-script entry `kairu` with three subcommands. `kairu serve` wraps `create_app` with full flag coverage (`--model`, `--host`, `--port`, `--cache-capacity`, `--adaptive-gamma`, `--max-prompt-chars`, `--max-tokens-cap`, `--request-timeout`, `--rate-limit`, `--rate-window`, `--redis`, `--log-level`); `kairu bench` re-exports `kairu.bench.main`; `kairu version` prints `__version__`. The `--redis` flag swaps in `RedisBackend` via `redis.asyncio` (lazy imported, only required when used).
- `kairu._hf_backend.HuggingFaceKVCachedModel` — drop-in subclass of `HuggingFaceModel` that retains `past_key_values` across calls, computes the longest-common-prefix between successive token-id lists, and feeds only the divergent suffix to the model. Exposes `kv_cache_stats` (`kv_hits`, `kv_misses`, `kv_hit_rate`, `cached_prefix_len`) and `reset_cache()`. Respects `model.config.max_position_embeddings` as the hard cap and resets cleanly on overflow.
- `Dockerfile` (multi-stage, slim, non-root user uid 1001, healthcheck against `/health`) and `.dockerignore`.
- `.github/workflows/docker.yml` — multi-arch (linux/amd64, linux/arm64) GHCR publish on push to `main` and on `v*.*.*` tags. Buildx + QEMU + GHA cache. PRs from forks are smoke-built without push (no credentials surfaced). Post-push smoke test runs `kairu version` against the freshly pushed image.
- 27 new tests across `tests/test_rate_limit.py` (12 — both backends, mock-redis pipeline replays the real semantics including speculative-add rollback), `tests/test_metrics_export.py` (8 — counter/gauge/histogram math, label escaping, exposition format), `tests/test_cli.py` (7 — argparse coverage of every flag, version subcommand, dispatch). `tests/test_server.py` gains 2 tests covering `/metrics` shape and 429 accounting.

### Changed

- `pyproject.toml` — `version: 0.5.0 → 0.6.0`; new `redis` optional extra (`redis>=5.0.0`); new `[project.scripts] kairu = "kairu.cli:main"`.
- `kairu/__init__.py` — exports `MetricsCollector`, `InMemoryBackend`, `RedisBackend`, `RateLimiterBackend`; `RateLimiter` now lives in `kairu.rate_limit` and is re-exported.
- `kairu/server.py` — refactored to use `kairu.rate_limit.RateLimiter`; `create_app` accepts `rate_limit_backend` and `metrics` kwargs; FastAPI app version pinned to `0.6.0`. Inline `RateLimiter`/`_Bucket` classes removed (BC-preserving — public name still resolves through `kairu.RateLimiter`).
- `kairu/_hf_backend.py` — `__all__` now lists `HuggingFaceKVCachedModel`.

### Architecture Decisions

- **Atomic Redis pipeline over Lua/EVAL.** A two-round-trip pipeline (`MULTI`+`EXEC`, then conditional `ZREM` on rejection) is portable across Redis 5+/Valkey, debuggable from `redis-cli`, and ~5–8 % slower than the equivalent Lua script in benchmarks. We pay that for now; an `EVAL` script can drop in if a real workload pushes back.
- **No `prometheus_client` dependency.** The exposition format is a 3-line-per-series text spec; rolling our own keeps the install graph clean and removes the multiprocessing helper that conflicts with how SSE servers run inside uvicorn workers.
- **HF KV cache at the `next_token_logits` boundary, not `generate()`.** The decoder modules already pass token-id prefixes through that one method; tracking the longest common prefix at that seam means `SpeculativeDecoder`, `LayerwiseEarlyExitDecoder`, and the streaming server all benefit from a single point change with no per-decoder integration work.
- **Docker image runs as uid 1001.** Cloud platforms enforcing pod-security policies (GKE Autopilot, EKS Pod Security Standards 'restricted') reject root containers; shipping non-root by default avoids that footgun.

---

## [0.5.0] — 2026-05-02

### Added — Model-Aware Optimization

- `kairu.layered.LayeredModelInterface` — `ModelInterface` extension exposing `num_layers()` and `layer_logits(token_ids, layer_idx)`. Mirrors HF transformers' `output_hidden_states=True` contract at the abstract-interface level.
- `kairu.layered.MockLayeredModel` — deterministic L-layer mock; logits sharpen monotonically with depth via `layer_logits = base_logits * (l / L)`. Drop-in for testing without ML deps.
- `kairu.layered.LayerwiseEarlyExitDecoder` — walks layers `[min_layer, num_layers]`, emits the argmax token at the first layer whose top-prob ≥ threshold. Reports per-token `exit_layers`, `mean_exit_layer`, and `compute_saved = 1 - mean_exit_layer / num_layers`.
- `kairu.kv_cache.LogitsCache` — bounded LRU cache (`OrderedDict.move_to_end`-backed); O(1) get/put; reports `hits`, `misses`, `evictions`, `size`, `hit_rate`.
- `kairu.kv_cache.CachedModel` — `ModelInterface` decorator that memoizes `next_token_logits` keyed by `tuple(token_ids)`. Drop-in for any model. Recycles target-model calls across speculative verification rounds and across repeated generations on the same prompt prefix.
- `kairu.gamma_scheduler.DynamicGammaScheduler` — AIMD scheduler over speculative γ. Math: `E[accepted] = (1 - ρ^(γ+1)) / (1 - ρ)` (Leviathan et al. 2023) → grow γ on high rolling acceptance, multiplicatively shrink on low. Configurable bounds, thresholds, window, increase/decrease rates.
- `kairu.auto_profile.AutoProfile.recommend(model, name_hint?, has_draft=False)` → frozen `DecoderProfile{strategy, gamma, early_exit_threshold, temperature, use_cache, cache_capacity, rationale}`. Strategy ∈ {`vanilla`, `early_exit`, `layered_early_exit`, `speculative`}. Decision rules: draft-name hint → vanilla; large vocab + draft → speculative (γ=4 below 100k vocab, 6 above); layered model → layered_early_exit; mid-size single backbone → early_exit; tiny → vanilla.
- `kairu.speculative.SpeculativeDecoder` — new optional `scheduler: DynamicGammaScheduler` arg; per-round `scheduler.update(round_accepted, gamma)`; stats include `final_gamma` and `gamma_adjustments` when the scheduler is wired.
- `kairu.wrapper.ModelWrapper` — new `cache_capacity: int = 0` flag (transparently wraps target + draft in `CachedModel`) and `adaptive_gamma: bool = False` flag (auto-attaches a `DynamicGammaScheduler`).
- 39 new tests across `tests/test_layered.py` (11), `tests/test_kv_cache.py` (9), `tests/test_gamma_scheduler.py` (10), `tests/test_auto_profile.py` (9). Cover: dtype/shape contracts, monotonic-confidence-with-depth, LRU eviction order, hit/miss math, cache helps repeat-prompt speculative calls, AIMD growth/shrink/clamp, mid-band hold, every config-validation path, AutoProfile dispatch on every branch, profile immutability and determinism.

### Changed

- `pyproject.toml` — `version: 0.4.0 → 0.5.0`; description expanded.
- `kairu/__init__.py` — version bump; new exports `AutoProfile`, `DecoderProfile`, `CachedModel`, `LogitsCache`, `DynamicGammaScheduler`, `LayeredModelInterface`, `LayerwiseEarlyExitDecoder`, `MockLayeredModel`.
- `kairu/speculative.py` — `from __future__ import annotations` added (3.10 PEP 604 union syntax was breaking 3.9 collection).
- `README.md` — adds the model-aware optimization section.

### Architecture Decisions

- **Logits memoization > per-layer KV tensor caching** at this abstraction tier. Real KV caches store per-layer K/V along the sequence dimension; that requires a backend-specific contract (HF `past_key_values`, MLX `make_kv_cache`, etc.). At the `ModelInterface` boundary the only observable is `next_token_logits(prefix)`, and *that* is a pure function of the prefix for a deterministic model. Caching at this layer is provably equivalent in I/O behavior and works uniformly across every backend the package supports today and tomorrow.
- **AIMD over learned γ scheduling.** A learned scheduler would need per-model calibration and a training pass. AIMD reaches a stable operating point in `O(log γ_max)` updates, has zero hyperparameter coupling to the model family, and matches the control law that is empirically known to converge cleanly under non-stationary acceptance rates (TCP congestion control, RED, ABR streaming).
- **AutoProfile is a deterministic heuristic, not a learned classifier.** The decision surface is small (4 strategies × a handful of features). A heuristic with explicit `rationale` strings is debuggable, reproducible, and zero-dependency. When/if the matrix grows past hand-encoding, swap in a tiny logistic regression — but not before.

---

## [0.4.0] — 2026-05-01

### Added

- `kairu.server.create_app(model?, tokenizer?, config?)` — FastAPI factory; returns an app exposing `POST /generate` (SSE) and `GET /health`. Lazy-imports FastAPI so `import kairu` stays cheap on environments without the `server` extra.
- `kairu.server.ServerConfig` — dataclass with `model_name`, `max_prompt_chars` (default 8192), `max_tokens_cap` (512), `request_timeout_s` (30.0), `rate_limit_requests` (10), `rate_limit_window_s` (10.0). Every limit is enforced at the API boundary before the tokenizer sees the request (CLAUDE.md §Inference Server Security).
- `kairu.server.RateLimiter` — pure-stdlib sliding-window per-key rate limiter; `asyncio.Lock`-guarded; mathematically: allow at time *t* iff `|{u ∈ window : t - u ≤ W}| < N`.
- `POST /generate` SSE stream — OpenAI `chat.completion.chunk`-compatible frames (`id`, `object`, `created`, `model`, `choices[].delta.content`, `choices[].finish_reason`) with a `kairu` extension carrying `token_id`, `index`, `latency_ms`, `tokens_per_s` per token, and `tokens_generated`/`total_s` on the final frame; terminates with `data: [DONE]\n\n` (OpenAI convention). `finish_reason` is one of `length`, `stop`, or `timeout`.
- Per-request wall-clock timeout enforcement via deadline check in the streaming generator (no partial frames lost — current chunk completes, then the stream ends with `finish_reason="timeout"`).
- Boundary input validation: prompt length, control-character rejection (`\x00-\x08\x0b\x0c\x0e-\x1f\x7f`), `max_tokens` cap, `temperature` ∈ [0, 2], non-negative `stop_token_id`.
- Privacy-preserving logging — only a SHA-256 prefix of the prompt is logged, never the raw content (CLAUDE.md "never log raw user prompt content at INFO level").
- 14 server tests in `tests/test_server.py` — health endpoint, OpenAI-compatible chunk shape, `[DONE]` sentinel, empty/oversized/control-char prompt rejection, max-tokens cap, temperature range, 429 rate limiting, stop-token short-circuit, request timeout `finish_reason="timeout"`, sliding-window unit tests, total_s monotonicity vs. summed per-token latencies.

### Changed

- `pyproject.toml` — `version: 0.3.0 → 0.4.0`; new `[project.optional-dependencies] server = ["fastapi>=0.110.0", "uvicorn>=0.27.0", "pydantic>=2.0.0"]`; `dev` extras gain `pytest-asyncio>=0.23.0` and `httpx>=0.27.0`; `[tool.pytest.ini_options]` now sets `asyncio_mode = "auto"`.
- `kairu/__init__.py` — guarded re-export of `create_app`, `ServerConfig`, `RateLimiter`; version bumped `0.3.0 → 0.4.0`.
- `README.md` — adds the `POST /generate` quickstart and the `kairu[server]` install hint.

### Architecture Decisions

- The Pydantic `GenerateRequest` schema lives at module scope (not inside `create_app`) so FastAPI's `typing.get_type_hints` can resolve the annotation during route registration. Same fix is applied to `Request` (imported from `starlette.requests` at module level rather than inside the factory) — closure-scoped FastAPI annotations are silently misclassified as query parameters.
- `_enforce_limits(req, cfg)` runs *after* Pydantic schema validation. The schema carries permissive caps (1M-char prompts, 100k tokens) and the per-instance `ServerConfig` tightens them. This decouples schema definition (one-time, module-load) from server config (per-instance, runtime).
- The SSE generator uses `await asyncio.sleep(0)` between tokens to force the StreamingResponse to flush each frame instead of buffering the whole generation. Without this, ASGI servers may coalesce frames and the "streaming" claim is a lie on the wire.
- Rate-limit state is in-process and per-key; horizontal scaling will require an external store (Redis) — flagged as a Phase 5 concern.

---

## [0.3.0] — 2026-04-28

### Added

- `kairu.bench.BenchmarkRunner` — drives N generation runs against any `ModelInterface` via `StreamingDecoder`; discards warmup runs; returns a `BenchmarkResult` with full latency statistics
- `kairu.bench.BenchmarkResult` — dataclass holding `latencies_s` (warmup-excluded per-run totals), `p50`/`p95`/`p99`/`mean`/`stddev` (seconds), `tokens_per_s_mean`, `hardware` dict, and ISO timestamp; `to_json()` and `save(base_dir)` — never overwrites existing files
- `kairu.bench._collect_hardware()` — hostname, OS/release/machine, CPU model (via `sysctl -n machdep.cpu.brand_string` on macOS, `/proc/cpuinfo` on Linux), total RAM (via `psutil` → `sysctl hw.memsize` → `/proc/meminfo`), Python version
- `kairu.bench.build_parser()` + `kairu.bench.main()` — CLI entry point for `python -m kairu.bench`; flags: `--model`, `--tokens`, `--runs`, `--warmup`, `--output`, `--name`; `--model mock` works with zero ML dependencies
- `kairu/__main__bench.py` — thin re-export shim so `from kairu.__main__bench import main` works in tests
- 8 benchmark tests in `tests/test_bench.py`: shape validation, p50≤p95≤p99 ordering, JSON round-trip with required keys, `save()` writes valid JSON with hardware, `_collect_hardware()` required keys, CLI exits 0, `--help` exits 0, saved filename contains timestamp + name
- `benchmarks/results/` directory auto-created on first `save()` call

### Changed

- `kairu/__init__.py` — exports `BenchmarkRunner`, `BenchmarkResult`; version bumped `0.2.0 → 0.3.0`

### Architecture Decisions

- Percentile computation uses pure `statistics` module + manual sorted-list linear interpolation — no scipy dependency
- `stddev` uses `statistics.stdev` (sample std dev, n-1); falls back to 0.0 when fewer than 2 samples
- Hardware collection is fully defensive — every platform call is wrapped in `try/except`; the function never raises
- CLI logic lives in `bench.py` (triggered by `-m kairu.bench`); `__main__bench.py` is a re-export shim for clean test imports

---

## [0.2.0] — 2026-04-28

### Added

- `kairu.tokenizer.TokenizerBase` — abstract tokenizer interface (`encode`, `decode`, `vocab_size`)
- `kairu.tokenizer.MockTokenizer` — deterministic mock tokenizer: space-splits text, hashes words to token IDs via `hash(w) % vocab_size`; zero dependencies; fully offline
- `kairu.tokenizer.HFTokenizer` — `TokenizerBase` wrapper around HF `AutoTokenizer`; defers import to construction; sets `pad_token = eos_token` when missing
- `kairu.streaming.StreamingDecoder` — NumPy-only greedy/temperature iterator over any `ModelInterface`; `stream()` yields token IDs one at a time with optional stop-token; `generate()` collects to list; seeded RNG (`default_rng(42)`) ensures reproducibility
- `kairu._hf_backend.HuggingFaceModel` — full rewrite: `encode(text)`, `decode(ids)`, `stream_generate(prompt, …)` via HF `TextIteratorStreamer` + background thread; all torch/transformers imports deferred to `__init__` so the module loads cleanly in Python-only environments; `torch_dtype=float32` (fixes broken-numpy compatibility on this machine)
- `kairu/__init__.py` — exports `StreamingDecoder`, `MockTokenizer`, `TokenizerBase`; guarded optional re-export of `HFTokenizer`
- 8 tokenizer tests in `tests/test_tokenizer.py` — fully offline, no ML deps
- 8 streaming tests in `tests/test_streaming.py` — `MockModel` only, no HF required
- 8 HF backend tests in `tests/test_hf_backend.py` — 4 structural/offline, 4 integration gated behind `KAIRU_TEST_HF=1`

### Changed

- `kairu/__init__.py` version bumped `0.1.0 → 0.2.0`
- `kairu._hf_backend.HuggingFaceModel` — `torch_dtype` changed `float16 → float32` to avoid broken-numpy `.numpy()` call failing on torch 2.2.x with numpy ABI mismatch

### Architecture Decisions

- `StreamingDecoder` is framework-free on purpose. Token-level streaming for a `MockModel` or any future custom backend does not need HF machinery; `TextIteratorStreamer` is only used inside `HuggingFaceModel.stream_generate`.
- All torch/transformers imports remain strictly inside `__init__` of classes that require them, keeping the module importable with zero ML libraries installed.
- HF integration tests are gated (`KAIRU_TEST_HF=1`) to keep CI fast and dependency-free by default.

---

## [0.1.0] — 2026-04-28

### Added

- `kairu.base.ModelInterface` — abstract contract (`vocab_size`, `next_token_logits`, `max_seq_len`) decoupling all engine logic from ML frameworks
- `kairu.mock_model.MockModel` — deterministic LCG-seeded mock implementing `ModelInterface`; enables full test coverage with no ML dependencies
- `kairu.speculative.SpeculativeDecoder` — draft-model speculative decoding with acceptance-ratio rejection sampling (Chen et al., 2023); configurable `gamma` lookahead and temperature; returns `(tokens, stats)` with acceptance rate
- `kairu.early_exit.EarlyExitDecoder` — halts generation when top-token probability exceeds `confidence_threshold` **or** Shannon entropy drops below `entropy_floor`; returns `(tokens, stats)` with exit reason
- `kairu.budget.TokenBudget` — hard cap on prompt + generated tokens; `consume(n)` returns actual tokens allowed; `remaining`, `exhausted`, `utilization()` properties
- `kairu.metrics.GenerationMetrics` — wall-clock generation timing, tokens/s, mean latency per token, speculative acceptance rate; `to_dict()` serialization
- `kairu.dashboard.KairuDashboard` — Rich `Live` context manager rendering a real-time metrics panel; `attach(metrics)` + `update()` API
- `kairu.wrapper.ModelWrapper` — unified wrapper wiring speculative or early-exit decoder with budget enforcement and metrics collection
- `kairu.wrapper.wrap_model()` — convenience factory accepting a `ModelInterface` or HF model name string; falls back to `MockModel` when `torch`/`transformers` are absent
- `kairu._hf_backend.HuggingFaceModel` — optional HF causal-LM backend (enabled via `kairu[hf]`)
- `pyproject.toml` — `hatchling` build, `requires-python = ">=3.10"`, `dev` + `hf` optional extras
- `.github/workflows/ci.yml` — GitHub Actions CI on push/PR to `main`; Python 3.11; `pytest -v`
- 31 tests across `test_budget`, `test_metrics`, `test_speculative`, `test_wrapper`; all passing

### Architecture Decisions

- Core engine is intentionally framework-free. All algorithm logic (speculative sampling, entropy gating, budget arithmetic) runs on NumPy only. HF/torch is a thin optional adapter.
- `SpeculativeDecoder` seeds its RNG at construction (`seed=42`) so tests are fully deterministic without mocking.
- `MockModel.next_token_logits` uses a Knuth multiplicative hash of the token-id sum as the RNG seed, giving context-dependent but reproducible logit distributions.
