# Kairu — Project Roadmap

> 流 · *to flow, to stream*

Current version: **v0.1.0**

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

## Phase 2 — HuggingFace Integration (v0.2.0)

- Full HF backend with streaming generation via `model.generate(..., streamer=...)`
- Tokenizer integration: `encode(text) → list[int]`, `decode(ids) → str`
- End-to-end test with a tiny model (e.g. `sshleifer/tiny-gpt2`) in CI
- `wrap_model("model-name")` no longer falls back to MockModel when `[hf]` is installed

---

## Phase 3 — Benchmarking (v0.3.0)

- `kairu.bench` module: `BenchmarkRunner` class
- p50 / p95 / p99 latency reporting across N generations
- CSV + JSON result export to `benchmarks/results/<timestamp>_<name>.json`
- Hardware metadata capture (platform, CPU, RAM, OS)
- CLI: `python -m kairu.bench --model mock --tokens 100 --runs 50`

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
