import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { Dial, ease } from "@konjoai/ui";
import { fetchSpeedup } from "../lib/api";
import type { SpeedupResult } from "../lib/types";

export interface SpeedupGaugeProps {
  rho?: number;
  gamma?: number;
}

/**
 * Speedup calculator: closed-form formula from Leviathan et al. 2023.
 * Renders as an animated dial showing expected speedup for given ρ and γ.
 */
export function SpeedupGauge({ rho = 0.78, gamma = 4 }: SpeedupGaugeProps) {
  const [result, setResult] = useState<SpeedupResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  useEffect(() => {
    setLoading(true);
    (async () => {
      const { result, fromMock } = await fetchSpeedup(rho, gamma);
      setResult(result);
      setFromMock(fromMock);
      setLoading(false);
    })();
  }, [rho, gamma]);

  if (!result) {
    return (
      <div className="glass-konjo rounded-konjo-lg p-5 flex items-center justify-center h-48">
        <p className="text-konjo-fg-muted text-konjo-mono text-[12px]">loading…</p>
      </div>
    );
  }

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Speedup Calculator
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Closed-form formula ·{" "}
            <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5">
        <div className="grid sm:grid-cols-[auto_1fr] gap-5 items-center">
          <motion.div
            key={`${rho}-${gamma}`}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, ease: ease.kanjo }}
          >
            <Dial
              value={result.speedup}
              min={1}
              max={Math.max(10, result.speedup + 2)}
              unit="×"
              label="Speedup"
              severity={result.speedup > 2 ? "ok" : "warn"}
              format={(v) => v.toFixed(2)}
              size={150}
              sublabel={loading ? "computing…" : "formula"}
            />
          </motion.div>

          <div className="space-y-3 min-w-0">
            <div className="space-y-1">
              <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
                Parameters
              </div>
              <div className="text-konjo-display text-[14px] text-konjo-fg font-mono">
                ρ = {rho.toFixed(2)} · γ = {gamma}
              </div>
            </div>

            <div className="pt-2 border-t border-konjo-line/40 space-y-2">
              <div>
                <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-1">
                  Formula
                </div>
                <div className="text-konjo-mono text-[11px] text-konjo-fg bg-konjo-surface rounded p-2">
                  <code>{result.derivation}</code>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-konjo-line/40">
              <p className="text-konjo-fg-muted text-[12px]">
                Expected speedup for ρ={rho.toFixed(2)} and γ={gamma} tokens lookahead.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
