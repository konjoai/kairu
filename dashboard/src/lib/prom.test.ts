import { describe, it, expect } from "vitest";
import { parseProm, summarize, bucketQuantile } from "./prom";
import { getMockPromText } from "./mock";

describe("parseProm", () => {
  it("parses counters with labels", () => {
    const m = parseProm(`# TYPE x counter
x{a="1"} 5
x{a="2"} 7
`);
    const x = m.get("x");
    expect(x?.kind).toBe("counter");
    if (x?.kind === "counter") {
      expect(x.samples.length).toBe(2);
      const total = x.samples.reduce((a, s) => a + s.value, 0);
      expect(total).toBe(12);
    }
  });

  it("parses gauges", () => {
    const m = parseProm(`# TYPE g gauge
g 3.14
`);
    const g = m.get("g");
    expect(g?.kind).toBe("gauge");
    if (g?.kind === "gauge") expect(g.samples[0].value).toBeCloseTo(3.14);
  });

  it("parses histograms with buckets, sum, count", () => {
    const m = parseProm(`# TYPE h histogram
h_bucket{le="0.001"} 1
h_bucket{le="0.01"} 5
h_bucket{le="+Inf"} 10
h_sum 0.5
h_count 10
`);
    const h = m.get("h");
    expect(h?.kind).toBe("histogram");
    if (h?.kind === "histogram") {
      expect(h.buckets.length).toBe(3);
      expect(h.buckets[2].le).toBe("+Inf");
      expect(h.sum).toBe(0.5);
      expect(h.count).toBe(10);
    }
  });
});

describe("summarize", () => {
  it("rolls kairu metrics into the cockpit shape", () => {
    const m = parseProm(getMockPromText());
    const s = summarize(m);
    expect(s.tokens_total.length).toBe(6732);
    expect(s.tokens_total.stop).toBe(1108);
    expect(s.active_streams).toBe(1);
    expect(s.token_latency_count).toBe(7840);
    expect(s.token_latency_buckets.length).toBeGreaterThan(0);
  });
});

describe("bucketQuantile", () => {
  const buckets = [
    { leMs: 1,   count: 100 },
    { leMs: 10,  count: 800 },
    { leMs: 50,  count: 950 },
    { leMs: 100, count: 990 },
    { leMs: Infinity, count: 1000 },
  ];

  it("returns the boundary at p50", () => {
    expect(bucketQuantile(buckets, 0.5)).toBe(10); // 800/1000 ≥ 0.5 → 10ms
  });
  it("returns the p95 bucket", () => {
    expect(bucketQuantile(buckets, 0.95)).toBe(50); // 950/1000 ≥ 0.95
  });
  it("returns null when total=0", () => {
    expect(bucketQuantile([{ leMs: 1, count: 0 }], 0.5)).toBeNull();
  });
});
