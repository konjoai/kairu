import { useEffect, useRef, useState } from "react";
import { KonjoApp } from "@konjoai/ui";
import { GenerationStream } from "./views/GenerationStream";
import { ThroughputCockpit } from "./views/ThroughputCockpit";
import { LatencyHistogram } from "./views/LatencyHistogram";
import { TokenLatencyChart } from "./views/TokenLatencyChart";
import { SpeculativeRace } from "./views/SpeculativeRace";
import { FinishReasonsRing } from "./views/FinishReasonsRing";
import { PromptBar } from "./views/PromptBar";
import { MetaInspector } from "./views/MetaInspector";
import { generateStream, fetchMetrics, fetchHealth } from "./lib/api";
import { summarize } from "./lib/prom";
import type {
  CockpitMetrics,
  FinishReason,
  GenerationToken,
} from "./lib/types";

const EMPTY_METRICS: CockpitMetrics = {
  tokens_total: { length: 0, stop: 0, timeout: 0, other: 0 },
  active_streams: 0,
  uptime_seconds: 0,
  request_count: 0,
  rate_limited: 0,
  errors_total: 0,
  token_latency_buckets: [],
  token_latency_count: 0,
  token_latency_sum_s: 0,
};

export default function App() {
  const [prompt, setPrompt] = useState<string>("Explain speculative decoding in two sentences.");
  const [maxTokens, setMaxTokens] = useState<number>(48);
  const [temperature, setTemperature] = useState<number>(0.85);

  const [streamState, setStreamState] = useState<"idle" | "streaming" | "done" | "error">("idle");
  const [tokens, setTokens] = useState<GenerationToken[]>([]);
  const [finishReason, setFinishReason] = useState<FinishReason | undefined>();
  const [totalSeconds, setTotalSeconds] = useState<number | undefined>();
  const [model, setModel] = useState<string>("kairu");
  const [version, setVersion] = useState<string | undefined>();
  const [generationFromMock, setGenerationFromMock] = useState<boolean>(false);
  const [speculativeFromMock] = useState<boolean>(false);

  const [metrics, setMetrics] = useState<CockpitMetrics>(EMPTY_METRICS);

  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const h = await fetchHealth();
      if (cancelled) return;
      if (h.model) setModel(h.model);
      if (h.version) setVersion(h.version);
    })();
    const refresh = async () => {
      const { map } = await fetchMetrics();
      if (cancelled) return;
      setMetrics(summarize(map));
    };
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const startGenerate = () => {
    cancelRef.current?.();
    setTokens([]);
    setFinishReason(undefined);
    setTotalSeconds(undefined);
    setStreamState("streaming");
    const handle = generateStream(
      { prompt, max_tokens: maxTokens, temperature },
      (t, opts) => {
        setGenerationFromMock(!!opts.fromMock);
        setTokens((arr) => [...arr, t]);
      },
    );
    cancelRef.current = handle.cancel;
    handle.done.then((res) => {
      setFinishReason(res.finishReason);
      setTotalSeconds(res.totalSeconds);
      if (res.model) setModel(res.model);
      setGenerationFromMock(res.fromMock);
      setStreamState("done");
    }).catch(() => setStreamState("error"));
  };

  useEffect(() => {
    startGenerate();
    return () => cancelRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const live = streamState === "streaming";
  const finalized = streamState === "done";

  return (
    <KonjoApp
      product="kairu"
      tagline="Speed Cockpit · real-time inference made visible"
      status={
        live ? { label: "streaming", severity: "info" } :
        finalized ? { label: generationFromMock ? "offline · mocks" : "live", severity: generationFromMock ? "warn" : "ok" } :
                    { label: "idle", severity: "info" }
      }
    >
      <Hero />

      <div className="space-y-6 mt-10">
        <PromptBar
          prompt={prompt}
          onPromptChange={setPrompt}
          maxTokens={maxTokens}
          onMaxTokensChange={setMaxTokens}
          temperature={temperature}
          onTemperatureChange={setTemperature}
          onSubmit={startGenerate}
          disabled={live}
          submitLabel={live ? "generating…" : finalized ? "regenerate" : "generate"}
        />

        <section className="grid lg:grid-cols-[1fr_360px] gap-4 items-start">
          <GenerationStream
            tokens={tokens}
            streaming={live}
            finishReason={finishReason}
            totalSeconds={totalSeconds}
            prompt={prompt}
          />
          <ThroughputCockpit tokens={tokens} streaming={live} />
        </section>

        <section className="grid lg:grid-cols-2 gap-4">
          <TokenLatencyChart tokens={tokens} />
          <LatencyHistogram metrics={metrics} />
        </section>

        <SpeculativeRace rho={0.78} gamma={4} autoPlay />

        <section className="grid lg:grid-cols-[auto_1fr] gap-4 items-start">
          <FinishReasonsRing metrics={metrics} />
          <MetaInspector
            model={model}
            version={version}
            metrics={metrics}
            generationFromMock={generationFromMock}
            speculativeFromMock={speculativeFromMock}
          />
        </section>

        <Footer />
      </div>
    </KonjoApp>
  );
}

function Hero() {
  return (
    <section className="text-center pt-6 pb-2">
      <p className="text-konjo-mono uppercase tracking-[0.32em] text-konjo-violet" style={{ fontSize: 11 }}>
        kairu · 海流 · the current
      </p>
      <h1
        className="text-konjo-display text-konjo-fg mt-4 mx-auto"
        style={{ fontSize: 52, fontWeight: 600, letterSpacing: "-0.025em", maxWidth: 920, lineHeight: 1.05 }}
      >
        Inference, <span style={{ color: "var(--color-konjo-accent)" }}>watched live</span>.
      </h1>
      <p
        className="text-konjo-fg-muted mt-5 mx-auto"
        style={{ fontSize: 16, maxWidth: 640, lineHeight: 1.55 }}
      >
        Every token. Every millisecond. Every speculative draft accepted or rejected. Speculative decoding turns parallelism into seconds saved — kairu shows you exactly where.
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer
      className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-fg-muted text-konjo-mono"
      style={{ fontSize: 12 }}
    >
      <div className="flex flex-wrap gap-4 justify-between items-baseline">
        <span>
          built on{" "}
          <span className="text-konjo-fg">@konjoai/ui</span>
          {" · "}
          <span className="text-konjo-fg">/generate</span>
          {" · "}
          <span className="text-konjo-fg">/metrics</span>
          {" · "}
          <span className="text-konjo-fg">/api/simulate-race</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio · vectro · squish · kyro · miru · kohaku · toki · squash
        </span>
      </div>
    </footer>
  );
}
