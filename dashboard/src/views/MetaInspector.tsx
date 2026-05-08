import type { CockpitMetrics } from "../lib/types";

export interface MetaInspectorProps {
  model: string;
  version?: string;
  metrics: CockpitMetrics;
  /** Source of the live generation pane. */
  generationFromMock: boolean;
  /** Source of the speculative race pane. */
  speculativeFromMock: boolean;
}

function StatBlock({
  label, value, accent,
}: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2 rounded-konjo bg-konjo-surface/60 border border-konjo-line/60">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
        {label}
      </div>
      <div
        className="text-konjo-mono tabular-nums text-konjo-fg"
        style={{ fontSize: 13, color: accent ?? "var(--color-konjo-fg)" }}
      >
        {value}
      </div>
    </div>
  );
}

function fmtUptime(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d`;
}

export function MetaInspector({ model, version, metrics, generationFromMock, speculativeFromMock }: MetaInspectorProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
      <StatBlock
        label="model"
        value={`${model}${version ? ` · v${version}` : ""}`}
        accent={generationFromMock ? "var(--color-konjo-warm)" : "var(--color-konjo-fg)"}
      />
      <StatBlock
        label="generation"
        value={generationFromMock ? "mock" : "live"}
        accent={generationFromMock ? "var(--color-konjo-warm)" : "var(--color-konjo-good)"}
      />
      <StatBlock
        label="speculative"
        value={speculativeFromMock ? "simulated · offline" : "simulated · server"}
        accent="var(--color-konjo-violet)"
      />
      <StatBlock
        label="active streams"
        value={`${metrics.active_streams}`}
        accent={metrics.active_streams > 0 ? "var(--color-konjo-good)" : "var(--color-konjo-fg)"}
      />
      <StatBlock
        label="uptime"
        value={metrics.uptime_seconds > 0 ? fmtUptime(metrics.uptime_seconds) : "—"}
      />
      <StatBlock
        label="errors / rate-limited"
        value={`${metrics.errors_total} / ${metrics.rate_limited}`}
        accent={metrics.errors_total + metrics.rate_limited > 0 ? "var(--color-konjo-warm)" : "var(--color-konjo-fg-muted)"}
      />
    </div>
  );
}
