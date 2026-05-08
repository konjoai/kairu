# Changelog

All notable changes to `@kairu/dashboard` are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/).

## [0.1.0] — 2026-05-08

### Added — Sprint 3: Speed Cockpit

The flagship cinematic UI for kairu. Watch a model decode in real time, every token painted with its own latency.

- **Repository scaffold** — Vite 8 + React 19 + TypeScript + Tailwind v4 + Vitest 4. Consumes `@konjoai/ui` via `file:../../konjoai-ui`. React and motion are deduped at the resolver to share one singleton.

- **Eight views**:
  - [`<PromptBar>`](./src/views/PromptBar.tsx) — prompt + max_tokens + temperature controls; ⌘/ctrl-Enter to generate.
  - [`<GenerationStream>`](./src/views/GenerationStream.tsx) — tokens stream in via the Konjo `<TokenStream>` primitive, painted with per-token latency hue (cool=fast, hot=slow). Final frame summary shows tokens · total_s · finish_reason.
  - [`<ThroughputCockpit>`](./src/views/ThroughputCockpit.tsx) — live tok/s dial + per-token sparkline.
  - [`<TokenLatencyChart>`](./src/views/TokenLatencyChart.tsx) — per-token latency line chart with p50/p95 reference bands.
  - [`<LatencyHistogram>`](./src/views/LatencyHistogram.tsx) — Prometheus `kairu_token_latency_seconds` histogram bars + p50/p95/p99 derived from cumulative buckets.
  - [`<SpeculativeRace>`](./src/views/SpeculativeRace.tsx) — cinematic accept/reject waterfall driven by `/api/simulate-race` (offline numpy-backed simulation; the live `/generate` does not yet emit per-draft accept flags).
  - [`<FinishReasonsRing>`](./src/views/FinishReasonsRing.tsx) — donut of cumulative tokens by `finish_reason` (length / stop / timeout / other).
  - [`<MetaInspector>`](./src/views/MetaInspector.tsx) — model · version · generation source (live vs mock) · speculative source · active streams · uptime · errors.

- **Library layer**:
  - [`types.ts`](./src/lib/types.ts) — TS mirrors of `/generate` + `/metrics` + `/api/simulate-race` shapes.
  - [`sse.ts`](./src/lib/sse.ts) — SSE/NDJSON parser. Auto-detects format. Handles `[DONE]` sentinel, comment keepalives, mid-frame buffer continuation.
  - [`prom.ts`](./src/lib/prom.ts) — Prometheus exposition parser. Counters · gauges · histograms with labels. `summarize()` rolls metrics into the cockpit shape; `bucketQuantile()` extracts p50/p95/p99 from cumulative buckets.
  - [`api.ts`](./src/lib/api.ts) — `generateStream` (SSE consumer with onToken callback) · `fetchMetrics` · `simulateRace` · `fetchHealth`. Each transparently falls back to mocks when its server is unreachable.
  - [`mock.ts`](./src/lib/mock.ts) — `buildMockGeneration` (24-token sample with realistic latency noise), `buildMockSimulateRace` (implements the Leviathan et al. speedup formula `(1 - ρ^(γ+1)) / (1 - ρ)`), `getMockPromText` (full Prometheus exposition).

- **Honest visualization**:
  - **Speculative accept/reject is simulated**, never fabricated into the real generation pane. Today's `/generate` doesn't emit per-draft accept flags; the SpeculativeRace lane drives off `/api/simulate-race` and the MetaInspector reports `speculative: simulated`. A future backend lift could wire `SpeculativeDecoder` into `/generate` and emit per-token `accepted` flags with no dashboard surface change.
  - **Per-token latency is real.** Every dot in TokenLatencyChart and every hue in GenerationStream comes from the wire-format `kairu.latency_ms` field — no synthesis.

- **Tests** — 29 Vitest cases covering: SSE/NDJSON frame splitting (with comment keepalives + mid-frame continuation), Prometheus parsing (counters / gauges / histograms with labels), histogram quantile derivation, mock-fixture invariants (Leviathan formula at ρ=1, γ=4 → 5×), behavioral tests for `<PromptBar>` / `<GenerationStream>` / `<MetaInspector>`. All green.

- **Docs** — README, CLAUDE.md (operating rules), this changelog.

### Notes

- Sprint 3 of the 10-sprint Konjo UI Initiative.
- All animation respects `prefers-reduced-motion`.
- Two transports, two ports: `/generate` /metrics /health → kairu FastAPI (8000); `/api/simulate-race` → demo stdlib server (7777). Vite dev proxy routes them transparently.
