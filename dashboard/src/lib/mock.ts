/**
 * Mock fixtures for offline development. Used when /generate and the demo
 * server are both unreachable.
 */
import type {
  GenerationToken,
  GenerationResult,
  SimulateRaceRound,
  SimulateRaceResult,
} from "./types";

const SAMPLE_TOKENS = [
  "Speculative", " decoding", " accelerates", " inference", " by", " having", " a",
  " smaller", " draft", " model", " propose", " tokens", " in", " parallel", ",",
  " then", " verifying", " them", " in", " a", " single", " forward", " pass", ".",
];

export function buildMockGeneration(): GenerationResult {
  const t0 = performance.now ? performance.now() : Date.now();
  const tokens: GenerationToken[] = SAMPLE_TOKENS.map((text, index) => {
    const noise = Math.sin(index * 1.3) * 4 + Math.cos(index * 2.1) * 2;
    const latencyMs = 18 + noise + (index < 4 ? 6 : 0);
    return {
      index,
      text,
      tokenId: 1000 + index,
      latencyMs,
      tokensPerS: ((index + 1) / Math.max(0.001, (index + 1) * latencyMs / 1000)),
      arrivedAt: t0 + (index + 1) * latencyMs,
    };
  });
  const total_s = tokens.reduce((a, t) => a + t.latencyMs, 0) / 1000;
  return {
    tokens,
    finishReason: "stop",
    totalSeconds: total_s,
    model: "kairu-mock",
    fromMock: true,
  };
}

/** Pacing helper for replaying the mock generation as if it were live. */
export function mockPacing(t: GenerationToken): number {
  return Math.max(8, Math.min(140, t.latencyMs));
}

/** Synthetic speculative race when /api/simulate-race is unreachable. */
export function buildMockSimulateRace(rho = 0.78, gamma = 4): SimulateRaceResult {
  const rounds: SimulateRaceRound[] = [];
  let total = 0;
  let r = 1;
  while (total < 32 && r < 50) {
    const proposed = gamma;
    const accepted_mask: boolean[] = [];
    let accepted = 0;
    for (let i = 0; i < proposed; i++) {
      const ok = Math.random() < rho;
      accepted_mask.push(ok);
      if (ok) accepted++;
      else break; // speculative chain breaks on first reject
    }
    const emitted = accepted + 1; // +1 for the verified-correct token from the target
    total += emitted;
    rounds.push({
      round: r,
      gamma_used: gamma,
      draft_tokens_proposed: proposed,
      accepted_mask: padMask(accepted_mask, proposed),
      accepted_count: accepted,
      tokens_emitted_this_round: emitted,
      running_total_tokens: total,
      scheduler_gamma_after: gamma,
    });
    r++;
  }
  // Leviathan et al.: E[T] = (1 - rho^(γ+1)) / (1 - rho)
  const expected_speedup = rho === 1 ? gamma + 1 : (1 - Math.pow(rho, gamma + 1)) / (1 - rho);
  return { rho, gamma, rounds, total_tokens: total, expected_speedup };
}

function padMask(m: boolean[], gamma: number): boolean[] {
  const out = m.slice();
  while (out.length < gamma) out.push(false);
  return out.slice(0, gamma);
}

const MOCK_PROM_TEXT = `# HELP kairu_requests_total Total HTTP requests received by endpoint and status.
# TYPE kairu_requests_total counter
kairu_requests_total{endpoint="/generate",status="200"} 142
kairu_requests_total{endpoint="/health",status="200"} 45
kairu_requests_total{endpoint="/metrics",status="200"} 12
kairu_requests_total{endpoint="/generate",status="422"} 2

# HELP kairu_tokens_generated_total Total tokens emitted by /generate.
# TYPE kairu_tokens_generated_total counter
kairu_tokens_generated_total{finish_reason="length"} 6732
kairu_tokens_generated_total{finish_reason="stop"} 1108
kairu_tokens_generated_total{finish_reason="timeout"} 0

# HELP kairu_errors_total Total errors broken down by kind.
# TYPE kairu_errors_total counter
kairu_errors_total{kind="validation"} 1
kairu_errors_total{kind="invalid_json"} 1

# HELP kairu_rate_limited_total Total requests rejected by the rate limiter.
# TYPE kairu_rate_limited_total counter
kairu_rate_limited_total 0

# HELP kairu_active_streams Number of in-flight /generate SSE streams.
# TYPE kairu_active_streams gauge
kairu_active_streams 1

# HELP kairu_token_latency_seconds Per-token generation latency.
# TYPE kairu_token_latency_seconds histogram
kairu_token_latency_seconds_bucket{le="0.001"} 342
kairu_token_latency_seconds_bucket{le="0.005"} 4128
kairu_token_latency_seconds_bucket{le="0.01"} 6480
kairu_token_latency_seconds_bucket{le="0.025"} 7524
kairu_token_latency_seconds_bucket{le="0.05"} 7696
kairu_token_latency_seconds_bucket{le="0.1"} 7810
kairu_token_latency_seconds_bucket{le="0.25"} 7839
kairu_token_latency_seconds_bucket{le="0.5"} 7840
kairu_token_latency_seconds_bucket{le="1.0"} 7840
kairu_token_latency_seconds_bucket{le="+Inf"} 7840
kairu_token_latency_seconds_sum 287.34
kairu_token_latency_seconds_count 7840

# HELP kairu_process_uptime_seconds Process uptime in seconds.
# TYPE kairu_process_uptime_seconds gauge
kairu_process_uptime_seconds 3847.23
`;

export function getMockPromText(): string { return MOCK_PROM_TEXT; }
