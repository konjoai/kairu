# kairu · Speed Cockpit

A flagship Konjo UI for **kairu** — the real-time LLM inference optimizer.

> 海流 (kairu) — ocean current · 急流 — rapid · 流れ — flow

Watch a model decode in real time. Every token is painted with its latency. The throughput dial breathes with the stream. The speculative race shows accept/reject decisions as they fire. The Prometheus histogram resolves into p50 · p95 · p99.

## Quick start

```bash
npm install
npm run dev      # → http://localhost:5176
npm test         # vitest (29 tests)
npm run build    # production build → dist/
```

To wire the dashboard to a live kairu backend:

```bash
# Terminal 1 — start the FastAPI server (port 8000) for /generate + /metrics + /health
cd /Users/wesleyscholl/kairu
uvicorn kairu.server:app --port 8000

# Terminal 2 — start the demo server (port 7777) for /api/simulate-race
python demo/server.py

# Terminal 3 — start the dashboard (proxies /generate /metrics /health → :8000, /api → :7777)
cd dashboard
npm run dev
```

When either server is unreachable the dashboard transparently falls back to mocks. The MetaInspector reports the source of every pane (`live` vs `offline · mocks`) so you always know what you're looking at.

## Stack

`React 19` · `TypeScript` · `Vite 8` · `Tailwind CSS v4` · `motion` · `Vitest`
Built on top of [`@konjoai/ui`](../../konjoai-ui) — the shared design system for the KonjoAI portfolio.

## What you'll see

| Panel               | What it shows                                                          |
|---------------------|------------------------------------------------------------------------|
| **Hero**            | The kairu promise · cyan / accent gradient                             |
| **PromptBar**       | Prompt + max_tokens + temperature controls · ⌘/ctrl-Enter to generate  |
| **GenerationStream**| Tokens stream in with per-token latency hue (cool=fast, hot=slow)      |
| **ThroughputCockpit**| Live tok/s dial + per-token sparkline                                 |
| **TokenLatencyChart**| Per-token latency line chart with p50/p95 reference bands             |
| **LatencyHistogram**| Prometheus bucket distribution + p50/p95/p99 from `/metrics`           |
| **SpeculativeRace** | Cinematic accept/reject waterfall · driven by `/api/simulate-race`     |
| **FinishReasonsRing**| Donut of cumulative tokens by finish_reason (length/stop/timeout)     |
| **MetaInspector**   | Model · version · live vs mock · active streams · uptime · errors      |

## Architecture

- **Two transports** — `/generate` (real model, FastAPI on :8000) and `/api/simulate-race` (offline numpy simulation on the demo server :7777). The dashboard treats them as different lanes; speculative state is **never fabricated** in the real generation pane.

- **SSE / NDJSON parser** ([src/lib/sse.ts](./src/lib/sse.ts)). Auto-detects format. Handles SSE `[DONE]` sentinel, comment keepalives, mid-frame buffer continuation, and newline-delimited NDJSON.

- **Prometheus parser** ([src/lib/prom.ts](./src/lib/prom.ts)). Tiny exposition-format parser sufficient for kairu's specific output. Computes p50/p95/p99 directly from cumulative buckets.

- **Mock-first design** ([src/lib/mock.ts](./src/lib/mock.ts)). Every transport has a hand-crafted fallback so the dashboard is always shippable. Mocks honor the same shape as the real wire format — same parser path, just different bytes.

## Configuration

- `VITE_KAIRU_API` — base URL of the kairu FastAPI server (default: `""`, leans on the Vite dev proxy).
- `VITE_KAIRU_DEMO_API` — base URL of the kairu demo server (default: `""`).
- The dev server proxies:
  - `/generate`, `/metrics`, `/health` → `http://localhost:8000`
  - `/api/*` → `http://localhost:7777`

## Tests

```bash
npm test
```

Covers: SSE/NDJSON frame splitting (incl. comment keepalives, mid-frame), Prometheus parsing (counters/gauges/histograms with labels), histogram quantile derivation, mock-fixture invariants (Leviathan speedup formula), and behavioral tests for `<PromptBar>`, `<GenerationStream>`, `<MetaInspector>`. 29 tests, all green.

## Honesty notes

- **Speculative accept/reject is simulated**, not measured. Today's `/generate` doesn't emit per-draft accept flags. The SpeculativeRace pane drives off the demo server's offline numpy simulation; the MetaInspector flags this with `speculative: simulated`. A future backend lift could wire `SpeculativeDecoder` into `/generate` and emit per-token `accepted` flags; the dashboard would consume them with no surface change.

- **Per-token latency is real.** Every dot in TokenLatencyChart and every hue in GenerationStream comes from the `kairu.latency_ms` field on the wire — no synthesis.

See [`CLAUDE.md`](./CLAUDE.md) for operating rules.
