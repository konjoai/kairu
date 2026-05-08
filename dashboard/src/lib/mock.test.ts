import { describe, it, expect } from "vitest";
import { buildMockGeneration, buildMockSimulateRace, mockPacing } from "./mock";

describe("buildMockGeneration", () => {
  it("returns a sensible token sequence with finishReason 'stop'", () => {
    const g = buildMockGeneration();
    expect(g.tokens.length).toBeGreaterThan(10);
    expect(g.finishReason).toBe("stop");
    expect(g.fromMock).toBe(true);
  });
  it("token.index is 0..N-1 contiguously", () => {
    const g = buildMockGeneration();
    g.tokens.forEach((t, i) => expect(t.index).toBe(i));
  });
  it("each token reports a positive latencyMs", () => {
    const g = buildMockGeneration();
    for (const t of g.tokens) expect(t.latencyMs).toBeGreaterThan(0);
  });
});

describe("buildMockSimulateRace", () => {
  it("computes the Leviathan speedup formula at ρ=1, γ=4 → 5x", () => {
    const r = buildMockSimulateRace(1, 4);
    expect(r.expected_speedup).toBeCloseTo(5, 5);
  });
  it("emits at least one round and the running total grows monotonically", () => {
    const r = buildMockSimulateRace(0.78, 4);
    expect(r.rounds.length).toBeGreaterThan(0);
    let prev = 0;
    for (const round of r.rounds) {
      expect(round.running_total_tokens).toBeGreaterThanOrEqual(prev);
      prev = round.running_total_tokens;
    }
  });
  it("each accepted_mask has γ entries", () => {
    const r = buildMockSimulateRace(0.5, 4);
    for (const round of r.rounds) expect(round.accepted_mask.length).toBe(4);
  });
});

describe("mockPacing", () => {
  it("clamps pacing into [8, 140]", () => {
    expect(mockPacing({ index: 0, text: "x", tokenId: 0, latencyMs: 1, tokensPerS: 0, arrivedAt: 0 })).toBe(8);
    expect(mockPacing({ index: 0, text: "x", tokenId: 0, latencyMs: 9999, tokensPerS: 0, arrivedAt: 0 })).toBe(140);
  });
});
