/**
 * TypeScript types mirroring kairu's API contract.
 * Source of truth lives in /Users/wesleyscholl/kairu/kairu/server.py.
 */

export interface GenerateRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  stop_token_id?: number;
}

/** Per-token kairu extension carried inside chat.completion.chunk frames. */
export interface KairuTokenMeta {
  token_id: number;
  index: number;
  latency_ms: number;
  tokens_per_s: number;
}

/** Final-frame kairu extension. */
export interface KairuFinalMeta {
  tokens_generated: number;
  total_s: number;
}

export type FinishReason = "length" | "stop" | "timeout" | "confidence" | "entropy";

export interface ChunkChoice {
  index: 0;
  delta: { content?: string };
  finish_reason: FinishReason | null;
}

export interface TokenChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChunkChoice[];
  kairu: KairuTokenMeta | KairuFinalMeta;
}

export type StreamState = "idle" | "streaming" | "done" | "error";

export interface GenerationToken {
  index: number;
  text: string;
  tokenId: number;
  latencyMs: number;
  tokensPerS: number;
  arrivedAt: number;     // performance.now() timestamp
}

export interface GenerationResult {
  tokens: GenerationToken[];
  finishReason?: FinishReason;
  totalSeconds?: number;
  model: string;
  fromMock: boolean;
}

/** Per-round speculative simulation result from /api/simulate-race. */
export interface SimulateRaceRound {
  round: number;
  gamma_used: number;
  draft_tokens_proposed: number;
  accepted_mask: boolean[];
  accepted_count: number;
  tokens_emitted_this_round: number;
  running_total_tokens: number;
  scheduler_gamma_after: number;
}

export interface SimulateRaceResult {
  rho: number;
  gamma: number;
  rounds: SimulateRaceRound[];
  total_tokens: number;
  expected_speedup: number;
}

/** Parsed Prometheus metric. */
export type PromMetric =
  | { kind: "counter"; name: string; help?: string; samples: PromSample[] }
  | { kind: "gauge";   name: string; help?: string; samples: PromSample[] }
  | { kind: "histogram"; name: string; help?: string; buckets: PromBucket[]; sum: number; count: number };

export interface PromSample {
  labels: Record<string, string>;
  value: number;
}

export interface PromBucket {
  le: number | "+Inf";
  count: number;
}

/** A bucketed view of /metrics relevant to the cockpit. */
export interface CockpitMetrics {
  tokens_total: { length: number; stop: number; timeout: number; other: number };
  active_streams: number;
  uptime_seconds: number;
  request_count: number;
  rate_limited: number;
  errors_total: number;
  /** [bucket boundary in ms, count] tuples, ordered. */
  token_latency_buckets: { leMs: number; count: number }[];
  token_latency_count: number;
  token_latency_sum_s: number;
}
