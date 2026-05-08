import { useMemo } from "react";
import { TokenStream, ease } from "@konjoai/ui";
import type { StreamToken } from "@konjoai/ui";
import { motion, AnimatePresence } from "motion/react";
import type { GenerationToken, FinishReason } from "../lib/types";

export interface GenerationStreamProps {
  tokens: GenerationToken[];
  streaming: boolean;
  finishReason?: FinishReason;
  totalSeconds?: number;
  prompt?: string;
}

/**
 * Live token stream with per-token latency hue. Each generated token's
 * background opacity = (slow vs fast), tinted hot for slow / cool for fast.
 *
 *   weight = clamp((latencyMs - p10) / (p90 - p10), 0, 1)
 *   colour = hot if slow, cool if fast
 *
 * The Konjo TokenStream primitive consumes the colour + weight directly.
 */
export function GenerationStream({ tokens, streaming, finishReason, totalSeconds, prompt }: GenerationStreamProps) {
  const stream = useMemo(() => buildStream(tokens), [tokens]);

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Generation
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Per-token latency painted onto the text · cool = fast · hot = slow
          </p>
        </div>
        <AnimatePresence>
          {finishReason && totalSeconds != null && (
            <motion.div
              initial={{ opacity: 0, x: 6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4, ease: ease.kanjo }}
              className="text-konjo-mono text-[11px] text-konjo-fg-muted"
            >
              <span className="text-konjo-fg">{tokens.length}</span> tokens ·{" "}
              <span className="text-konjo-fg">{totalSeconds.toFixed(2)}s</span> ·{" "}
              <span style={{ color: "var(--color-konjo-good)" }}>{finishReason}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {prompt && (
        <div className="glass-konjo rounded-konjo p-3 border-l-2" style={{ borderLeftColor: "var(--color-konjo-violet)" }}>
          <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-violet mb-1">
            prompt
          </div>
          <div className="text-konjo-fg-muted" style={{ fontSize: 13, lineHeight: 1.5 }}>{prompt}</div>
        </div>
      )}

      <TokenStream
        tokens={stream}
        cursor={streaming}
        maxHeight={260}
        font="display"
        density="loose"
        className="text-[15px]"
      />
    </section>
  );
}

function buildStream(tokens: GenerationToken[]): StreamToken[] {
  if (tokens.length === 0) return [];
  const lats = tokens.map((t) => t.latencyMs).sort((a, b) => a - b);
  const p10 = lats[Math.floor(lats.length * 0.1)];
  const p90 = lats[Math.floor(lats.length * 0.9)];
  const span = Math.max(0.001, p90 - p10);
  return tokens.map((t) => {
    const w = Math.max(0, Math.min(1, (t.latencyMs - p10) / span));
    // Colour interpolates between cool (low w) and hot (high w).
    const colour = w >= 0.66 ? "var(--color-konjo-hot)" :
                   w >= 0.33 ? "var(--color-konjo-warm)" :
                                "var(--color-konjo-accent)";
    return {
      id: t.index,
      text: t.text,
      color: colour,
      weight: 0.15 + w * 0.55,
    };
  });
}
