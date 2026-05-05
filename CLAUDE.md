# kairu

Real-time inference optimizer for LLMs — speculative decoding, early-exit decoding, KV cache management, token budgets, benchmarking (p50/p95/p99), streaming API, and a Rich live dashboard.

**v0.6.0** — Python-first, no ML frameworks required for core logic. HF integration optional.

## Stack
Python 3.9+ · NumPy · Rich · FastAPI (optional, `kairu[server]`) · uvicorn (optional) · transformers + PyTorch (optional, `kairu[hf]`) · hatchling

## Commands
```bash
python -m pytest tests/ -x                                              # full test suite (no ML deps)
KAIRU_TEST_HF=1 python -m pytest tests/                                 # include HF integration tests
python -m kairu.bench --model mock --tokens 100 --runs 50 --warmup 5   # benchmark
uvicorn kairu.server:app --reload                                        # streaming API (requires kairu[server])
```

## Critical Constraints
- No `unwrap()` — raise with a clear message or log + re-raise
- No silent failures — log a warning when a fallback path swallows an error
- HF/torch imports must be **deferred** to `__init__` — module must be importable without ML frameworks
- `HuggingFaceModel` integration tests are always gated behind `KAIRU_TEST_HF=1` — CI runs offline
- `BenchmarkResult.save()` must never overwrite an existing file — always append timestamp to filename
- Benchmark percentile computation uses pure `statistics` + sorted-list nearest-rank — no scipy dependency
- Hardware metadata keys (`hostname`, `os`, `cpu`, `ram_gb`, `python`) must all be present in every `BenchmarkResult`
- `StreamingDecoder` uses only `NumPy` + `ModelInterface` — never imports HF in the streaming hot path
- Version bumps touch `pyproject.toml` + `kairu/__init__.py`

## Module Map
| Module | Role |
|--------|------|
| `kairu/base.py` | `ModelInterface` ABC — zero-dependency contract for all backends |
| `kairu/mock_model.py` | `MockModel` — deterministic LCG-seeded mock |
| `kairu/speculative.py` | `SpeculativeDecoder` — draft-model lookahead with rejection sampling |
| `kairu/early_exit.py` | `EarlyExitDecoder` — confidence-threshold + entropy-floor halting |
| `kairu/streaming.py` | `StreamingDecoder` — greedy/temperature token-by-token iterator |
| `kairu/kv_cache.py` | KV cache management |
| `kairu/budget.py` | `TokenBudget` — hard prompt+generation cap |
| `kairu/bench.py` | `BenchmarkRunner` + `BenchmarkResult` — p50/p95/p99/stddev |
| `kairu/dashboard.py` | `KairuDashboard` — Rich live metric panel |
| `kairu/metrics.py` | `GenerationMetrics` — wall-clock, tok/s, acceptance rate |
| `kairu/server.py` | FastAPI streaming inference API (optional) |
| `kairu/cli.py` | CLI entry point |
| `kairu/_hf_backend.py` | `HuggingFaceModel` with `TextIteratorStreamer` (optional) |

## Planning Docs
- `PLAN.md` — current phase state and version history
- `CHANGELOG.md` — all notable changes

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
