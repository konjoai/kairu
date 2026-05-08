import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { CockpitMetrics } from "../lib/types";

export interface FinishReasonsRingProps {
  metrics: CockpitMetrics;
  size?: number;
}

const SLICE_COLOR: Record<string, string> = {
  length:  "var(--color-konjo-cool)",
  stop:    "var(--color-konjo-good)",
  timeout: "var(--color-konjo-hot)",
  other:   "var(--color-konjo-violet)",
};

/**
 * Donut chart of cumulative tokens emitted per finish_reason.
 */
export function FinishReasonsRing({ metrics, size = 160 }: FinishReasonsRingProps) {
  const t = metrics.tokens_total;
  const total = t.length + t.stop + t.timeout + t.other;
  const slices = total > 0
    ? [
        { label: "length",  value: t.length },
        { label: "stop",    value: t.stop },
        { label: "timeout", value: t.timeout },
        { label: "other",   value: t.other },
      ].filter((s) => s.value > 0)
    : [{ label: "—", value: 1 }];

  const cx = size / 2;
  const cy = size / 2;
  const r  = size / 2 - 12;
  const stroke = size / 12;
  const C = 2 * Math.PI * r;

  let acc = 0;

  return (
    <div className="glass-konjo rounded-konjo-lg p-5 flex items-center gap-5">
      <div style={{ width: size, height: size, position: "relative" }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-konjo-line)" strokeWidth={stroke} opacity={0.4} />
          <g transform={`rotate(-90 ${cx} ${cy})`}>
            {slices.map((s, i) => {
              const frac = s.value / Math.max(1, total > 0 ? total : 1);
              const offset = -acc * C;
              acc += frac;
              const c = SLICE_COLOR[s.label] ?? "var(--color-konjo-fg-muted)";
              return (
                <motion.circle
                  key={s.label + i}
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill="none"
                  stroke={c}
                  strokeWidth={stroke}
                  strokeLinecap="butt"
                  strokeDasharray={`${frac * C} ${C}`}
                  strokeDashoffset={offset}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4, ease: ease.kanjo, delay: i * 0.1 }}
                  style={{ filter: `drop-shadow(0 0 5px ${c})` }}
                />
              );
            })}
          </g>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted text-[9px]">
            tokens
          </div>
          <div className="text-konjo-display text-konjo-fg leading-none mt-0.5 tabular-nums" style={{ fontSize: 22, fontWeight: 600 }}>
            {total > 0 ? compact(total) : "—"}
          </div>
          <div className="text-konjo-mono text-[10px] text-konjo-fg-muted mt-1">finish reasons</div>
        </div>
      </div>
      <div className="space-y-1.5 min-w-0">
        {(["length", "stop", "timeout", "other"] as const).map((k) => (
          <div key={k} className="flex items-center gap-2 text-konjo-mono text-[12px]">
            <span
              className="inline-block rounded-full"
              style={{ width: 7, height: 7, background: SLICE_COLOR[k], boxShadow: `0 0 6px ${SLICE_COLOR[k]}` }}
            />
            <span className="text-konjo-fg-muted" style={{ minWidth: 64 }}>{k}</span>
            <span className="text-konjo-fg tabular-nums">{compact(t[k])}</span>
            <span className="text-konjo-fg-faint tabular-nums" style={{ minWidth: 44, textAlign: "right" }}>
              {total > 0 ? `${((t[k] / total) * 100).toFixed(0)}%` : "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}
