# Changelog

All notable changes to Kairu follow [Conventional Commits](https://www.conventionalcommits.org/) and [Keep a Changelog](https://keepachangelog.com/) conventions.

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
