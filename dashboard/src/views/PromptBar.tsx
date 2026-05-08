import { useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";

export interface PromptBarProps {
  prompt: string;
  onPromptChange: (s: string) => void;
  maxTokens: number;
  onMaxTokensChange: (n: number) => void;
  temperature: number;
  onTemperatureChange: (n: number) => void;
  onSubmit: () => void;
  disabled: boolean;
  submitLabel?: string;
}

/**
 * Prompt input + max_tokens + temperature controls + Generate button.
 * Mirrors the kairu /generate request shape exactly.
 */
export function PromptBar({
  prompt, onPromptChange,
  maxTokens, onMaxTokensChange,
  temperature, onTemperatureChange,
  onSubmit, disabled,
  submitLabel = "generate",
}: PromptBarProps) {
  const [focused, setFocused] = useState(false);
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: ease.kanjo }}
      className="space-y-2"
    >
      <div
        className={[
          "glass-konjo rounded-konjo p-3",
          focused ? "shadow-konjo-glow" : "",
        ].join(" ")}
      >
        <textarea
          rows={2}
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !disabled) onSubmit();
          }}
          placeholder="prompt the model · ⌘/ctrl-enter to generate"
          className="w-full bg-transparent border-0 outline-none text-konjo-fg placeholder:text-konjo-fg-faint resize-none"
          style={{ fontSize: 14, lineHeight: 1.55 }}
        />
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <NumberControl
          label="max_tokens"
          value={maxTokens}
          onChange={onMaxTokensChange}
          min={1} max={512} step={1}
        />
        <NumberControl
          label="temperature"
          value={temperature}
          onChange={onTemperatureChange}
          min={0} max={2} step={0.05}
          format={(v) => v.toFixed(2)}
        />
        <div className="flex-1" />
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled || prompt.trim().length === 0}
          className={[
            "px-5 py-2 rounded-konjo text-konjo-mono uppercase tracking-[0.18em] text-[11px] transition-colors",
            disabled || prompt.trim().length === 0
              ? "bg-konjo-surface text-konjo-fg-faint cursor-not-allowed"
              : "bg-konjo-accent text-konjo-bg hover:brightness-110 cursor-pointer shadow-konjo-glow",
          ].join(" ")}
        >
          {submitLabel}
        </button>
      </div>
    </motion.div>
  );
}

function NumberControl({
  label, value, onChange, min, max, step, format,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min: number; max: number; step: number;
  format?: (v: number) => string;
}) {
  return (
    <label className="flex items-center gap-2 glass-konjo rounded-konjo px-3 py-1.5">
      <span className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted text-[10px]">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
        className="bg-transparent border-0 outline-none text-konjo-mono text-konjo-fg tabular-nums w-16 text-right"
        style={{ fontSize: 13 }}
      />
      {format && <span className="text-konjo-mono text-konjo-fg-faint text-[10px]">{format(value)}</span>}
    </label>
  );
}
