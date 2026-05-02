# Changelog

All notable changes to Kairu follow [Conventional Commits](https://www.conventionalcommits.org/) and [Keep a Changelog](https://keepachangelog.com/) conventions.

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
