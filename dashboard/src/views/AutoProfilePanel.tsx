import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import { fetchRecommend } from "../lib/api";
import type { ModelSpec, RecommendResult } from "../lib/types";

const PRESETS: Record<string, ModelSpec> = {
  "llama-3-8b": {
    model_name: "llama-3-8b",
    vocab_size: 128_256,
    has_draft: true,
    layered: true,
  },
  "gpt-4o-mini": {
    model_name: "gpt-4o-mini",
    vocab_size: 128_000,
    has_draft: true,
    layered: true,
  },
  "mistral-7b": {
    model_name: "mistral-7b",
    vocab_size: 32_000,
    has_draft: true,
    layered: true,
  },
  "phi-3.5": {
    model_name: "phi-3.5",
    vocab_size: 32_064,
    has_draft: false,
    layered: false,
  },
};

export interface AutoProfilePanelProps {}

/**
 * AutoProfile: feed model specs, get strategy recommendations.
 * Wraps /api/recommend with a form UI and results display.
 */
export function AutoProfilePanel({}: AutoProfilePanelProps) {
  const [spec, setSpec] = useState<ModelSpec>(PRESETS["llama-3-8b"]);
  const [result, setResult] = useState<RecommendResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  const handleRecommend = async () => {
    setLoading(true);
    const { result, fromMock } = await fetchRecommend(spec);
    setResult(result);
    setFromMock(fromMock);
    setLoading(false);
  };

  const handlePreset = (key: string) => {
    const preset = PRESETS[key];
    if (preset) {
      setSpec(preset);
      setResult(null);
    }
  };

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            AutoProfile
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Automatic strategy recommendation ·{" "}
            <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div className="space-y-2">
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
            Model Preset
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {Object.keys(PRESETS).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => handlePreset(k)}
                className={[
                  "px-3 py-2 rounded-konjo text-konjo-mono text-[11px] uppercase tracking-[0.14em] font-medium transition-colors",
                  spec.model_name === PRESETS[k].model_name
                    ? "bg-konjo-accent text-konjo-bg"
                    : "bg-konjo-surface border border-konjo-line text-konjo-fg-muted hover:text-konjo-fg",
                ].join(" ")}
              >
                {k}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
            Model Details
          </label>
          <div className="grid sm:grid-cols-2 gap-3">
            <div>
              <input
                type="text"
                value={spec.model_name}
                onChange={(e) => setSpec({ ...spec, model_name: e.target.value })}
                placeholder="Model name"
                className="w-full px-3 py-2 bg-konjo-surface border border-konjo-line rounded-konjo text-konjo-fg text-[13px] placeholder-konjo-fg-muted"
              />
            </div>
            <div>
              <input
                type="number"
                value={spec.vocab_size}
                onChange={(e) => setSpec({ ...spec, vocab_size: parseInt(e.target.value) || 0 })}
                placeholder="Vocab size"
                className="w-full px-3 py-2 bg-konjo-surface border border-konjo-line rounded-konjo text-konjo-fg text-[13px] placeholder-konjo-fg-muted"
              />
            </div>
            <label className="flex items-center gap-2 text-konjo-fg text-[13px]">
              <input
                type="checkbox"
                checked={spec.has_draft}
                onChange={(e) => setSpec({ ...spec, has_draft: e.target.checked })}
                className="rounded"
              />
              Has draft model
            </label>
            <label className="flex items-center gap-2 text-konjo-fg text-[13px]">
              <input
                type="checkbox"
                checked={spec.layered}
                onChange={(e) => setSpec({ ...spec, layered: e.target.checked })}
                className="rounded"
              />
              Layered architecture
            </label>
          </div>
        </div>

        <motion.button
          type="button"
          onClick={handleRecommend}
          disabled={loading}
          whileHover={!loading ? { scale: 1.02 } : undefined}
          whileTap={!loading ? { scale: 0.98 } : undefined}
          className="w-full px-4 py-2 bg-konjo-accent text-konjo-bg rounded-konjo text-[13px] font-medium uppercase tracking-[0.16em] disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >
          {loading ? "Analyzing…" : "Recommend Strategy"}
        </motion.button>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3, ease: ease.kanjo }}
              className="pt-4 border-t border-konjo-line/40 space-y-3"
            >
              <div>
                <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-1">
                  Recommended Strategy
                </div>
                <div className="text-konjo-display text-[18px] font-semibold text-konjo-accent">
                  {result.strategy}
                </div>
              </div>

              <div className="grid sm:grid-cols-2 gap-3 text-[12px] font-mono">
                <div className="bg-konjo-surface rounded p-2">
                  <div className="text-konjo-fg-muted mb-1">Lookahead (γ)</div>
                  <div className="text-konjo-fg">{result.gamma}</div>
                </div>
                <div className="bg-konjo-surface rounded p-2">
                  <div className="text-konjo-fg-muted mb-1">Early-exit threshold</div>
                  <div className="text-konjo-fg">{(result.early_exit_threshold * 100).toFixed(0)}%</div>
                </div>
                <div className="bg-konjo-surface rounded p-2">
                  <div className="text-konjo-fg-muted mb-1">Temperature</div>
                  <div className="text-konjo-fg">{result.temperature.toFixed(2)}</div>
                </div>
                <div className="bg-konjo-surface rounded p-2">
                  <div className="text-konjo-fg-muted mb-1">Use cache</div>
                  <div className="text-konjo-fg">{result.use_cache ? "Yes" : "No"}</div>
                </div>
              </div>

              {result.rationale && (
                <div className="bg-konjo-surface rounded p-3">
                  <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2">
                    Rationale
                  </div>
                  <p className="text-konjo-fg text-[12px] leading-relaxed">
                    {result.rationale}
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
