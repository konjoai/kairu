/**
 * Tiny Prometheus exposition-format parser.
 *
 * Handles counter, gauge, and histogram lines. Ignores summaries (kairu doesn't
 * emit any). Designed for kairu's specific /metrics output — not a full parser.
 */
import type { PromMetric, PromSample, CockpitMetrics } from "./types";

interface RawLine {
  name: string;
  labels: Record<string, string>;
  value: number;
}

const TYPE_LINE = /^# TYPE (\S+) (counter|gauge|histogram|summary|untyped)/;
const HELP_LINE = /^# HELP (\S+) (.*)/;

export function parseProm(text: string): Map<string, PromMetric> {
  const types = new Map<string, "counter" | "gauge" | "histogram">();
  const helps = new Map<string, string>();
  const rawByName = new Map<string, RawLine[]>();

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("#")) {
      const t = TYPE_LINE.exec(line);
      if (t) {
        const k = t[2] as "counter" | "gauge" | "histogram" | "summary" | "untyped";
        if (k === "counter" || k === "gauge" || k === "histogram") types.set(t[1], k);
        continue;
      }
      const h = HELP_LINE.exec(line);
      if (h) helps.set(h[1], h[2]);
      continue;
    }
    const parsed = parseSample(line);
    if (!parsed) continue;
    const arr = rawByName.get(parsed.name) ?? [];
    arr.push(parsed);
    rawByName.set(parsed.name, arr);
  }

  const out = new Map<string, PromMetric>();

  for (const [name, kind] of types) {
    const help = helps.get(name);
    if (kind === "counter" || kind === "gauge") {
      const rows = rawByName.get(name) ?? [];
      const samples: PromSample[] = rows.map((r) => ({ labels: r.labels, value: r.value }));
      out.set(name, { kind, name, help, samples });
    }
  }

  // Histograms: combine *_bucket / *_sum / *_count under a single base name.
  for (const [name, kind] of types) {
    if (kind !== "histogram") continue;
    const help = helps.get(name);
    const bucketRows = rawByName.get(`${name}_bucket`) ?? [];
    const sumRows    = rawByName.get(`${name}_sum`)    ?? [];
    const countRows  = rawByName.get(`${name}_count`)  ?? [];
    const buckets = bucketRows.map((r) => {
      const le = r.labels.le;
      return { le: le === "+Inf" ? "+Inf" as const : Number(le), count: r.value };
    });
    out.set(name, {
      kind: "histogram",
      name,
      help,
      buckets,
      sum: sumRows[0]?.value ?? 0,
      count: countRows[0]?.value ?? 0,
    });
  }

  return out;
}

function parseSample(line: string): RawLine | null {
  // name{label1="x",label2="y"} value
  const idx = line.indexOf("{");
  let name: string;
  let rest: string;
  let labels: Record<string, string> = {};
  if (idx === -1) {
    const sp = line.indexOf(" ");
    if (sp === -1) return null;
    name = line.slice(0, sp);
    rest = line.slice(sp + 1).trim();
  } else {
    name = line.slice(0, idx);
    const close = line.indexOf("}", idx);
    if (close === -1) return null;
    const inside = line.slice(idx + 1, close);
    rest = line.slice(close + 1).trim();
    labels = parseLabels(inside);
  }
  const v = Number(rest.split(/\s+/)[0]);
  if (!Number.isFinite(v)) return null;
  return { name, labels, value: v };
}

function parseLabels(s: string): Record<string, string> {
  const out: Record<string, string> = {};
  // simple splitter — kairu values don't contain quoted commas
  for (const pair of s.split(",")) {
    const eq = pair.indexOf("=");
    if (eq === -1) continue;
    const k = pair.slice(0, eq).trim();
    const v = pair.slice(eq + 1).trim().replace(/^"|"$/g, "");
    if (k) out[k] = v;
  }
  return out;
}

/**
 * Reduce a parsed Prometheus map to the fields the cockpit cares about.
 */
export function summarize(metrics: Map<string, PromMetric>): CockpitMetrics {
  const tokens: CockpitMetrics["tokens_total"] = { length: 0, stop: 0, timeout: 0, other: 0 };
  const tokenMetric = metrics.get("kairu_tokens_generated_total");
  if (tokenMetric && tokenMetric.kind === "counter") {
    for (const s of tokenMetric.samples) {
      const reason = s.labels.finish_reason;
      if (reason === "length") tokens.length += s.value;
      else if (reason === "stop") tokens.stop += s.value;
      else if (reason === "timeout") tokens.timeout += s.value;
      else tokens.other += s.value;
    }
  }

  const active = metrics.get("kairu_active_streams");
  const uptime = metrics.get("kairu_process_uptime_seconds");
  const reqs = metrics.get("kairu_requests_total");
  const errs = metrics.get("kairu_errors_total");
  const rl = metrics.get("kairu_rate_limited_total");

  const sumCounter = (m: PromMetric | undefined) =>
    m && m.kind === "counter" ? m.samples.reduce((a, s) => a + s.value, 0) : 0;
  const firstGauge = (m: PromMetric | undefined) =>
    m && m.kind === "gauge" ? m.samples[0]?.value ?? 0 : 0;

  const tlat = metrics.get("kairu_token_latency_seconds");
  const buckets: CockpitMetrics["token_latency_buckets"] = [];
  let tlatCount = 0;
  let tlatSum = 0;
  if (tlat && tlat.kind === "histogram") {
    for (const b of tlat.buckets) {
      const leMs = b.le === "+Inf" ? Infinity : b.le * 1000;
      buckets.push({ leMs, count: b.count });
    }
    tlatCount = tlat.count;
    tlatSum = tlat.sum;
  }

  return {
    tokens_total: tokens,
    active_streams: firstGauge(active),
    uptime_seconds: firstGauge(uptime),
    request_count: sumCounter(reqs),
    rate_limited: sumCounter(rl),
    errors_total: sumCounter(errs),
    token_latency_buckets: buckets,
    token_latency_count: tlatCount,
    token_latency_sum_s: tlatSum,
  };
}

/**
 * Approximate p_q from cumulative histogram buckets.
 * Returns the ms boundary of the first bucket whose cumulative count >= q*total.
 */
export function bucketQuantile(buckets: CockpitMetrics["token_latency_buckets"], q: number): number | null {
  if (buckets.length === 0) return null;
  const total = buckets[buckets.length - 1]?.count ?? 0;
  if (total === 0) return null;
  const target = total * Math.max(0, Math.min(1, q));
  for (const b of buckets) {
    if (b.count >= target) return Number.isFinite(b.leMs) ? b.leMs : Infinity;
  }
  return null;
}
