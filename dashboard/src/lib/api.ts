/**
 * kairu dashboard API client.
 *
 * Three transports:
 *   1. generateStream — POST /generate, parses SSE/NDJSON, emits per-token events
 *   2. fetchMetrics    — GET /metrics, returns parsed Prometheus map
 *   3. simulateRace    — POST /api/simulate-race (demo server)
 *
 * Each falls back transparently to mock fixtures when the relevant server
 * is unreachable. The dashboard is always demo-able offline.
 */
import type {
  GenerateRequest,
  GenerationToken,
  GenerationResult,
  TokenChunk,
  KairuTokenMeta,
  KairuFinalMeta,
  FinishReason,
  SimulateRaceResult,
  PromMetric,
  SpeedupResult,
  RecommendResult,
  ModelSpec,
} from "./types";
import { parseStreamChunk } from "./sse";
import { parseProm } from "./prom";
import { buildMockGeneration, buildMockSimulateRace, buildMockSpeedup, buildMockRecommend, mockPacing, getMockPromText } from "./mock";

const KAIRU_API = (import.meta.env.VITE_KAIRU_API as string | undefined) ?? "";
const DEMO_API  = (import.meta.env.VITE_KAIRU_DEMO_API as string | undefined) ?? "";

export interface GenerateStreamHandle {
  cancel: () => void;
  done: Promise<GenerationResult>;
}

/**
 * Stream tokens as they arrive. The promise resolves with a complete
 * GenerationResult on clean end (or a mock replay on failure).
 */
export function generateStream(
  req: GenerateRequest,
  onToken: (t: GenerationToken, opts: { fromMock: boolean }) => void,
): GenerateStreamHandle {
  const ctrl = new AbortController();
  let cancelled = false;
  const tokens: GenerationToken[] = [];

  const done = (async (): Promise<GenerationResult> => {
    try {
      const res = await fetch(KAIRU_API + "/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(req),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`http ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let model = "kairu";
      let finishReason: FinishReason | undefined;
      let totalSeconds: number | undefined;

      while (!cancelled) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        const { frames, rest } = parseStreamChunk(buf);
        buf = rest;
        for (const f of frames) {
          if (f.done) continue;
          const obj = parseChunk(f.json);
          if (!obj) continue;
          model = obj.model ?? model;
          if (isFinalChunk(obj)) {
            const km = obj.kairu as KairuFinalMeta;
            finishReason = obj.choices?.[0]?.finish_reason ?? undefined;
            totalSeconds = km.tokens_generated > 0 ? km.total_s : 0;
          } else {
            const km = obj.kairu as KairuTokenMeta;
            const tok: GenerationToken = {
              index: km.index,
              text: obj.choices?.[0]?.delta?.content ?? "",
              tokenId: km.token_id,
              latencyMs: km.latency_ms,
              tokensPerS: km.tokens_per_s,
              arrivedAt: performance.now(),
            };
            tokens.push(tok);
            onToken(tok, { fromMock: false });
          }
        }
      }

      return { tokens, finishReason, totalSeconds, model, fromMock: false };
    } catch (e) {
      if (cancelled) return { tokens, model: "kairu", fromMock: true };
      // Transparent mock replay
      const mock = buildMockGeneration();
      const start = performance.now();
      for (let i = 0; i < mock.tokens.length; i++) {
        if (cancelled) break;
        const t = mock.tokens[i];
        const stamped: GenerationToken = { ...t, arrivedAt: start + (i + 1) * mockPacing(t) };
        await sleep(mockPacing(t));
        if (cancelled) break;
        tokens.push(stamped);
        onToken(stamped, { fromMock: true });
      }
      return { ...mock, tokens };
    }
  })();

  return { cancel: () => { cancelled = true; ctrl.abort(); }, done };
}

function parseChunk(s: string): TokenChunk | null {
  try { return JSON.parse(s) as TokenChunk; } catch { return null; }
}

function isFinalChunk(c: TokenChunk): boolean {
  const fr = c.choices?.[0]?.finish_reason;
  return fr != null;
}

function sleep(ms: number): Promise<void> { return new Promise((r) => setTimeout(r, ms)); }

export async function fetchMetrics(): Promise<{ raw: string; map: Map<string, PromMetric>; fromMock: boolean }> {
  try {
    const res = await fetch(KAIRU_API + "/metrics");
    if (!res.ok) throw new Error(`http ${res.status}`);
    const raw = await res.text();
    return { raw, map: parseProm(raw), fromMock: false };
  } catch {
    const raw = getMockPromText();
    return { raw, map: parseProm(raw), fromMock: true };
  }
}

export async function simulateRace(rho: number, gamma: number, n_tokens = 32): Promise<{ result: SimulateRaceResult; fromMock: boolean }> {
  try {
    const res = await fetch(DEMO_API + "/api/simulate-race", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rho, gamma, n_tokens }),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const result = (await res.json()) as SimulateRaceResult;
    return { result, fromMock: false };
  } catch {
    return { result: buildMockSimulateRace(rho, gamma), fromMock: true };
  }
}

export async function fetchHealth(): Promise<{ ok: boolean; model?: string; version?: string }> {
  try {
    const res = await fetch(KAIRU_API + "/health");
    if (!res.ok) return { ok: false };
    const data = (await res.json()) as { status?: string; model?: string; version?: string };
    return { ok: data.status === "ok", model: data.model, version: data.version };
  } catch {
    return { ok: false };
  }
}

export async function fetchSpeedup(rho: number, gamma: number): Promise<{ result: SpeedupResult; fromMock: boolean }> {
  try {
    const res = await fetch(DEMO_API + "/api/speedup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rho, gamma }),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const result = (await res.json()) as SpeedupResult;
    return { result, fromMock: false };
  } catch {
    return { result: buildMockSpeedup(rho, gamma), fromMock: true };
  }
}

export async function fetchRecommend(spec: ModelSpec): Promise<{ result: RecommendResult; fromMock: boolean }> {
  try {
    const res = await fetch(DEMO_API + "/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const result = (await res.json()) as RecommendResult;
    return { result, fromMock: false };
  } catch {
    return { result: buildMockRecommend(spec), fromMock: true };
  }
}
