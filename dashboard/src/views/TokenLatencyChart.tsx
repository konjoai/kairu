import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { GenerationToken } from "../lib/types";

export interface TokenLatencyChartProps {
  tokens: GenerationToken[];
  height?: number;
}

/**
 * Per-token latency line chart. Each token is a dot at (index, latencyMs);
 * the line is gradient-coloured by its severity.
 */
export function TokenLatencyChart({ tokens, height = 140 }: TokenLatencyChartProps) {
  const W = 720;
  const H = height;
  const padX = 8;
  const padY = 12;
  const innerH = H - padY * 2;

  if (tokens.length === 0) {
    return (
      <div className="glass-konjo rounded-konjo-lg p-5 flex items-center justify-center text-konjo-fg-muted" style={{ height: H + 32 }}>
        <span className="text-konjo-mono text-[12px]">awaiting generation…</span>
      </div>
    );
  }

  const max = Math.max(1, ...tokens.map((t) => t.latencyMs));
  const min = 0;
  const stepX = tokens.length > 1 ? (W - padX * 2) / (tokens.length - 1) : 0;
  const yFor = (v: number) => padY + innerH - ((v - min) / Math.max(1, max - min)) * innerH;

  const path = tokens
    .map((t, i) => `${i === 0 ? "M" : "L"} ${padX + i * stepX} ${yFor(t.latencyMs)}`)
    .join(" ");

  // Median band
  const sorted = [...tokens].map((t) => t.latencyMs).sort((a, b) => a - b);
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? p50;

  return (
    <div className="glass-konjo rounded-konjo-lg p-5 space-y-3">
      <div>
        <div className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted text-[10px]">
          per-token latency · this run
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height: H }}>
        {/* Median band */}
        <line
          x1={padX} y1={yFor(p50)} x2={W - padX} y2={yFor(p50)}
          stroke="var(--color-konjo-good)" strokeDasharray="4 6" strokeOpacity={0.6} strokeWidth={1}
        />
        <line
          x1={padX} y1={yFor(p95)} x2={W - padX} y2={yFor(p95)}
          stroke="var(--color-konjo-warm)" strokeDasharray="4 6" strokeOpacity={0.6} strokeWidth={1}
        />
        <text x={W - padX - 30} y={yFor(p50) - 4} fill="var(--color-konjo-good)" style={{ fontFamily: "JetBrains Mono", fontSize: 9 }}>p50</text>
        <text x={W - padX - 30} y={yFor(p95) - 4} fill="var(--color-konjo-warm)" style={{ fontFamily: "JetBrains Mono", fontSize: 9 }}>p95</text>

        {/* Line */}
        <motion.path
          d={path}
          fill="none"
          stroke="var(--color-konjo-accent)"
          strokeWidth={1.6}
          strokeLinejoin="round"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.6, ease: ease.kanjo }}
        />

        {/* Dots */}
        {tokens.map((t, i) => {
          const x = padX + i * stepX;
          const y = yFor(t.latencyMs);
          const c = t.latencyMs <= p50 ? "var(--color-konjo-good)" :
                    t.latencyMs <= p95 ? "var(--color-konjo-warm)" :
                                          "var(--color-konjo-hot)";
          return <circle key={i} cx={x} cy={y} r={2.4} fill={c} style={{ filter: `drop-shadow(0 0 4px ${c})` }} />;
        })}
      </svg>
      <div className="flex justify-between text-konjo-mono text-[10px] text-konjo-fg-muted">
        <span>token 0</span>
        <span>token {tokens.length - 1}</span>
      </div>
    </div>
  );
}
