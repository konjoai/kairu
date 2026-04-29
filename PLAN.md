# Kairu — Project Roadmap

> 流 · *to flow, to stream*

Current version: **v0.2.0**

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

## Phase 4 — Streaming API (v0.4.0)

- FastAPI SSE endpoint: `POST /generate` → `text/event-stream`
- Per-token metrics emitted as SSE data frames
- OpenAI-compatible response format
- Rate limiting + timeout enforcement (CLAUDE.md security rules)
- Integration test with `httpx` async client

---

## Phase 5 — Model-Aware Optimization (v0.5.0)

- Architecture-specific early exit: hook into intermediate transformer layers
- KV-cache recycling across speculative verification steps
- Dynamic `gamma` scheduling: increase lookahead when acceptance rate is high
- `AutoProfile`: select best decoder strategy per model family
