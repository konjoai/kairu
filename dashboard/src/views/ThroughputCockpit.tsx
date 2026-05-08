import { useMemo } from "react";
import { motion } from "motion/react";
import { Dial, ease } from "@konjoai/ui";
import type { GenerationToken } from "../lib/types";

export interface ThroughputCockpitProps {
  tokens: GenerationToken[];
  streaming: boolean;
}

/**
 * Live throughput dial + a per-second sparkline. Both derive from the running
 * token stream — no metrics endpoint needed.
 */
export function ThroughputCockpit({ tokens, streaming }: ThroughputCockpitProps) {
  const { current, peak, sparkline } = useMemo(() => {
    if (tokens.length === 0) return { current: 0, peak: 0, sparkline: [] as number[] };
    const last = tokens[tokens.length - 1];
    const current = last.tokensPerS;
    let peak = 0;
    for (const t of tokens) if (t.tokensPerS > peak) peak = t.tokensPerS;
    // Build sparkline: per-token instantaneous tok/s, normalized later
    const inst: number[] = tokens.map((t) => t.latencyMs > 0 ? 1000 / t.latencyMs : 0);
    return { current, peak, sparkline: inst };
  }, [tokens]);

  // Pick a humane axis cap so the dial stays informative.
  const max = Math.max(60, Math.ceil(peak / 10) * 10);

  const sevForRate = current >= 50 ? "ok" : current >= 25 ? "info" : current >= 10 ? "warn" : "high";

  return (
    <div className="glass-konjo rounded-konjo-lg p-5 grid sm:grid-cols-[auto_1fr] gap-5 items-center">
      <div className="flex flex-col items-center gap-2">
        <Dial
          value={current}
          min={0}
          max={max}
          unit="tok/s"
          label="Throughput"
          severity={sevForRate}
          format={(v) => v.toFixed(1)}
          size={170}
          sublabel={streaming ? "live" : tokens.length === 0 ? "—" : "settled"}
        />
      </div>
      <div className="space-y-2 min-w-0">
        <div className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted text-[10px]">
          per-token tok/s · last {sparkline.length}
        </div>
        <Sparkline values={sparkline} />
        <div className="flex justify-between text-konjo-mono text-[11px] text-konjo-fg-muted">
          <span>peak <span className="text-konjo-fg">{peak.toFixed(1)}</span></span>
          <span>tokens <span className="text-konjo-fg">{tokens.length}</span></span>
        </div>
      </div>
    </div>
  );
}

function Sparkline({ values, height = 56 }: { values: number[]; height?: number }) {
  if (values.length === 0) {
    return (
      <div className="rounded-konjo bg-konjo-surface/60 border border-konjo-line/60" style={{ height }}>
        <div className="h-full flex items-center justify-center text-konjo-fg-faint text-konjo-mono text-[11px]">
          no data
        </div>
      </div>
    );
  }
  const W = 320;
  const H = height;
  const max = Math.max(...values, 1);
  const stepX = values.length > 1 ? W / (values.length - 1) : W;
  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"} ${i * stepX} ${H - (v / max) * (H - 4) - 2}`)
    .join(" ");
  return (
    <div className="rounded-konjo bg-konjo-surface/60 border border-konjo-line/60 px-2 py-1" style={{ height }}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
        <motion.path
          d={path}
          fill="none"
          stroke="var(--color-konjo-accent)"
          strokeWidth={1.6}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.6, ease: ease.kanjo }}
          style={{ filter: "drop-shadow(0 0 6px var(--color-konjo-glow-accent))" }}
        />
      </svg>
    </div>
  );
}
