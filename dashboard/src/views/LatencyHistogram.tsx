import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { bucketQuantile } from "../lib/prom";
import type { CockpitMetrics } from "../lib/types";

export interface LatencyHistogramProps {
  metrics: CockpitMetrics;
}

/**
 * Visualizes the kairu_token_latency_seconds histogram as a stepped chart.
 * The bottom row reports p50/p95/p99 derived from cumulative bucket counts.
 */
export function LatencyHistogram({ metrics }: LatencyHistogramProps) {
  const buckets = metrics.token_latency_buckets;
  const total = buckets[buckets.length - 1]?.count ?? 0;

  // Differential per-bucket counts for a more intuitive bar chart.
  const bars: { label: string; count: number; le: number }[] = [];
  let prev = 0;
  for (const b of buckets) {
    if (!Number.isFinite(b.leMs)) break;
    const count = Math.max(0, b.count - prev);
    prev = b.count;
    bars.push({ label: fmtMs(b.leMs), count, le: b.leMs });
  }
  const max = Math.max(1, ...bars.map((b) => b.count));

  const p50 = bucketQuantile(buckets, 0.5);
  const p95 = bucketQuantile(buckets, 0.95);
  const p99 = bucketQuantile(buckets, 0.99);
  const meanMs = total > 0 ? (metrics.token_latency_sum_s / total) * 1000 : 0;

  return (
    <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
      <div>
        <div className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted text-[10px]">
          token latency · /metrics histogram
        </div>
        <div
          className="text-konjo-display text-konjo-fg leading-none mt-1 tabular-nums"
          style={{ fontSize: 22, fontWeight: 600 }}
        >
          {total.toLocaleString()} <span className="text-konjo-fg-muted text-[13px] ml-1">tokens observed</span>
        </div>
      </div>

      <div className="flex items-end gap-1 h-24" role="list">
        {bars.map((b, i) => (
          <motion.div
            key={i}
            role="listitem"
            initial={{ scaleY: 0, opacity: 0 }}
            animate={{ scaleY: 1, opacity: 1 }}
            transition={{ duration: 0.45, ease: ease.kanjo, delay: i * 0.04 }}
            className="flex-1 origin-bottom rounded-sm relative group"
            style={{
              height: `${(b.count / max) * 100}%`,
              minHeight: 2,
              background: barColor(b.le),
              boxShadow: `0 0 6px ${barColor(b.le)}`,
            }}
            title={`≤ ${b.label} · ${b.count.toLocaleString()} tokens`}
          />
        ))}
      </div>
      <div className="flex justify-between text-konjo-mono text-[10px] text-konjo-fg-muted">
        {bars.length > 0 && <span>{bars[0].label}</span>}
        {bars.length > 0 && <span>{bars[bars.length - 1].label}</span>}
      </div>

      <div className="grid grid-cols-4 gap-2">
        <Stat label="mean"  value={fmtMs(meanMs)} accent="var(--color-konjo-accent)" />
        <Stat label="p50"   value={p50 != null ? fmtMs(p50) : "—"} accent="var(--color-konjo-good)" />
        <Stat label="p95"   value={p95 != null ? fmtMs(p95) : "—"} accent="var(--color-konjo-warm)" />
        <Stat label="p99"   value={p99 != null ? fmtMs(p99) : "—"} accent="var(--color-konjo-hot)" />
      </div>
    </div>
  );
}

function fmtMs(ms: number): string {
  if (!Number.isFinite(ms)) return "∞";
  if (ms < 1) return `${ms.toFixed(2)}ms`;
  if (ms < 10) return `${ms.toFixed(1)}ms`;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function barColor(leMs: number): string {
  if (leMs <= 5)   return "var(--color-konjo-good)";
  if (leMs <= 25)  return "var(--color-konjo-cool)";
  if (leMs <= 100) return "var(--color-konjo-warm)";
  return "var(--color-konjo-hot)";
}

function Stat({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded-konjo bg-konjo-surface/50 border border-konjo-line/60 px-2 py-1.5">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">{label}</div>
      <div className="text-konjo-mono tabular-nums" style={{ fontSize: 14, color: accent }}>{value}</div>
    </div>
  );
}
