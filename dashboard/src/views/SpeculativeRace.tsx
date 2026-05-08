import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Dial, ease } from "@konjoai/ui";
import { simulateRace } from "../lib/api";
import type { SimulateRaceResult } from "../lib/types";

export interface SpeculativeRaceProps {
  /** Acceptance rate ρ — 0..1. Default 0.78. */
  rho?: number;
  /** Draft lookahead γ — small int. Default 4. */
  gamma?: number;
  /** Auto-play once data is loaded. Default true. */
  autoPlay?: boolean;
}

/**
 * The cinematic accept/reject waterfall. Drives off /api/simulate-race
 * (offline numpy-backed simulation, never the live model) so what you
 * see here is provably the math, not the inference. The MetaInspector
 * upstream reports `speculative: simulated`.
 */
export function SpeculativeRace({ rho = 0.78, gamma = 4, autoPlay = true }: SpeculativeRaceProps) {
  const [result, setResult] = useState<SimulateRaceResult | null>(null);
  const [fromMock, setFromMock] = useState<boolean>(false);
  const [cursor, setCursor] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(autoPlay);

  // Fetch on mount + whenever ρ/γ change.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { result, fromMock } = await simulateRace(rho, gamma, 32);
      if (cancelled) return;
      setResult(result);
      setFromMock(fromMock);
      setCursor(0);
      if (autoPlay) setPlaying(true);
    })();
    return () => { cancelled = true; };
  }, [rho, gamma, autoPlay]);

  // Walk the cursor.
  useEffect(() => {
    if (!playing || !result) return;
    if (cursor >= result.rounds.length) { setPlaying(false); return; }
    const id = setTimeout(() => setCursor((c) => c + 1), 380);
    return () => clearTimeout(id);
  }, [playing, cursor, result]);

  const visibleRounds = result?.rounds.slice(0, cursor) ?? [];
  const speedup = result?.expected_speedup ?? 0;

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Speculative race
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            ρ={rho.toFixed(2)} · γ={gamma} · expected speedup{" "}
            <span style={{ color: "var(--color-kairu-fast, var(--color-konjo-good))" }}>{speedup.toFixed(2)}×</span>
            {" · "}
            <span className="text-konjo-fg-faint">{fromMock ? "offline mock" : "demo server"}</span>
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            if (cursor >= (result?.rounds.length ?? 0)) setCursor(0);
            setPlaying((p) => !p);
          }}
          disabled={!result}
          className={[
            "px-3 py-1.5 rounded-konjo text-konjo-mono uppercase tracking-[0.18em] text-[10px] transition-colors",
            playing
              ? "bg-konjo-surface text-konjo-fg-muted"
              : "bg-konjo-accent text-konjo-bg hover:brightness-110",
          ].join(" ")}
        >
          {playing ? "pause" : cursor >= (result?.rounds.length ?? 0) ? "replay" : "play"}
        </button>
      </header>

      <div className="grid lg:grid-cols-[1fr_auto] gap-4">
        <div className="glass-konjo rounded-konjo-lg p-5 min-h-[260px] overflow-hidden">
          <div className="flex flex-col gap-2">
            <AnimatePresence initial={false}>
              {visibleRounds.map((r) => (
                <motion.div
                  key={r.round}
                  initial={{ opacity: 0, y: 12, filter: "blur(4px)" }}
                  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.35, ease: ease.kanjo }}
                  className="flex items-center gap-2"
                >
                  <span
                    className="text-konjo-mono tabular-nums text-konjo-fg-muted text-[11px] shrink-0"
                    style={{ minWidth: 32 }}
                  >
                    r{String(r.round).padStart(2, "0")}
                  </span>
                  <div className="flex gap-1 flex-wrap">
                    {r.accepted_mask.map((ok, i) => (
                      <DraftPip key={i} accepted={ok} delay={i * 0.05} />
                    ))}
                    <VerifiedPip />
                  </div>
                  <span className="ml-auto text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted">
                    +{r.tokens_emitted_this_round} · Σ {r.running_total_tokens}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
            {visibleRounds.length === 0 && (
              <div className="text-konjo-fg-muted text-konjo-mono text-[12px] py-8 text-center">
                {result ? "press play to start the race" : "loading simulation…"}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col items-center justify-center gap-3 glass-konjo rounded-konjo-lg p-5">
          <Dial
            value={Math.min(speedup, 5)}
            min={1}
            max={5}
            unit="×"
            label="Speedup"
            severity="ok"
            format={(v) => v.toFixed(2)}
            size={150}
            sublabel={fromMock ? "Leviathan et al." : "live simulation"}
          />
          <Legend />
        </div>
      </div>
    </section>
  );
}

function DraftPip({ accepted, delay }: { accepted: boolean; delay: number }) {
  const c = accepted ? "var(--color-konjo-good)" : "var(--color-konjo-hot)";
  return (
    <motion.span
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.25, ease: ease.kanjo, delay }}
      className="inline-flex items-center justify-center text-konjo-mono text-[9px] uppercase tracking-[0.16em] tabular-nums"
      title={accepted ? "draft accepted" : "draft rejected"}
      style={{
        width: 22, height: 18,
        borderRadius: 4,
        background: accepted ? `color-mix(in oklch, ${c} 22%, transparent)` : `color-mix(in oklch, ${c} 18%, transparent)`,
        color: c,
        border: `1px solid ${c}`,
        boxShadow: `0 0 6px ${c}`,
      }}
    >
      {accepted ? "✓" : "✗"}
    </motion.span>
  );
}

function VerifiedPip() {
  return (
    <motion.span
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.3, ease: ease.kanjo, delay: 0.18 }}
      className="inline-flex items-center justify-center text-konjo-mono text-[9px] uppercase tracking-[0.16em]"
      title="target-model verified token"
      style={{
        width: 22, height: 18,
        borderRadius: 4,
        background: "color-mix(in oklch, var(--color-konjo-violet) 22%, transparent)",
        color: "var(--color-konjo-violet)",
        border: "1px solid var(--color-konjo-violet)",
        boxShadow: "0 0 8px var(--color-konjo-violet)",
      }}
    >
      ★
    </motion.span>
  );
}

function Legend() {
  return (
    <div className="text-konjo-mono text-[10px] uppercase tracking-[0.16em] text-konjo-fg-muted flex gap-3 flex-wrap justify-center">
      <span className="flex items-center gap-1.5">
        <span style={{ width: 8, height: 8, background: "var(--color-konjo-good)", borderRadius: 2, boxShadow: "0 0 6px var(--color-konjo-good)" }} />
        accept
      </span>
      <span className="flex items-center gap-1.5">
        <span style={{ width: 8, height: 8, background: "var(--color-konjo-hot)", borderRadius: 2, boxShadow: "0 0 6px var(--color-konjo-hot)" }} />
        reject
      </span>
      <span className="flex items-center gap-1.5">
        <span style={{ width: 8, height: 8, background: "var(--color-konjo-violet)", borderRadius: 2, boxShadow: "0 0 6px var(--color-konjo-violet)" }} />
        target
      </span>
    </div>
  );
}
