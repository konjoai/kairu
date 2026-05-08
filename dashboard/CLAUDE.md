# kairu/dashboard

Speed Cockpit — flagship cinematic UI for kairu. Vite + React + `@konjoai/ui`. Sprint 3 of the Konjo UI Initiative.

## Stack
React 19 · TypeScript · Vite 8 · Tailwind v4 (`@theme` config) · motion · Vitest 4 · `@konjoai/ui` (file: dep)

## Commands
```bash
npm install
npm run dev          # → http://localhost:5176 (proxies /generate /metrics /health → :8000, /api → :7777)
npm test             # vitest (29 tests)
npm run build        # tsc -b && vite build
npm run typecheck    # tsc -b --noEmit
```

## Critical Constraints
- React, react-dom, and motion are deduped in [vite.config.ts](./vite.config.ts) so the dashboard and `@konjoai/ui` share a singleton. Don't break that.
- `@konjoai/ui` is consumed via `file:../../konjoai-ui`. Tokens come from `@konjoai/ui/styles` — don't redefine.
- Two transports, two ports — keep them straight: `/generate` /metrics /health → kairu FastAPI (8000); `/api/simulate-race` → demo stdlib server (7777). The dashboard never conflates them.
- **Speculative accept/reject MUST come from `/api/simulate-race`, never synthesized into the live generation pane.** Today's `/generate` doesn't emit per-draft flags; drift the surface, not the truth. MetaInspector reports `speculative: simulated`.
- All 29 tests + the build must stay green.

## File Map
| Path | Role |
|------|------|
| `src/App.tsx` | Composition + generate state machine |
| `src/views/GenerationStream.tsx` | Real /generate tokens with per-token latency hue |
| `src/views/ThroughputCockpit.tsx` | tok/s dial + sparkline |
| `src/views/TokenLatencyChart.tsx` | Per-token latency line chart with p50/p95 bands |
| `src/views/LatencyHistogram.tsx` | Prometheus histogram bars + p50/p95/p99 |
| `src/views/SpeculativeRace.tsx` | Accept/reject waterfall via /api/simulate-race |
| `src/views/FinishReasonsRing.tsx` | Donut of cumulative tokens by finish_reason |
| `src/views/PromptBar.tsx` | Prompt + max_tokens + temperature + Generate |
| `src/views/MetaInspector.tsx` | Model · version · live vs mock · uptime · errors |
| `src/lib/types.ts` | TS mirrors of /generate + /metrics shapes |
| `src/lib/api.ts` | generateStream + fetchMetrics + simulateRace + fetchHealth |
| `src/lib/sse.ts` | SSE/NDJSON parser (auto-detect, keepalive-tolerant) |
| `src/lib/prom.ts` | Prometheus exposition parser + summarize + bucketQuantile |
| `src/lib/mock.ts` | MOCK fixtures + buildMockSimulateRace (Leviathan formula) |

## Backend integration
- `POST /generate` — SSE (default) or NDJSON (Accept: application/x-ndjson). OpenAI-compatible `chat.completion.chunk` frames with kairu extension `{token_id, index, latency_ms, tokens_per_s}`. Final frame `{tokens_generated, total_s, finish_reason}`.
- `GET /metrics` — Prometheus text format. Counters, gauges, histograms.
- `GET /health` — `{status, model, version}`.
- `POST /api/simulate-race` (demo server) — `{rho, gamma, n_tokens}` → `{rounds, total_tokens, expected_speedup}` per round with `accepted_mask`.
- Both servers' CORS is open. Vite dev proxy is for ergonomic relative paths only.

## When extending
- New panel? Lives in `src/views/`. Always ship a Vitest test.
- New backend shape? Mirror types in [src/lib/types.ts](./src/lib/types.ts), add a mock fixture, then add the API method to [src/lib/api.ts](./src/lib/api.ts) with a mock fallback.
- New design token? Add to `@konjoai/ui` (so all flagships inherit), not here.
- Future backend lift: when `SpeculativeDecoder` is wired into `/generate`, emit per-token `accepted` boolean and `exit_layer` int. The dashboard already has the consumer slots (TokenLatencyChart, GenerationStream); only the wire format changes.

## Sprint context
This is **Sprint 3** of the 10-sprint Konjo UI Initiative. Sprint 0 = `@konjoai/ui` foundation. Sprint 1 = squash Compliance Bridge. Sprint 2 = miru Mind of the Machine. Sprint 4 = squish Inference Cockpit (next).
