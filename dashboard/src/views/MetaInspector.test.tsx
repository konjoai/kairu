import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetaInspector } from "./MetaInspector";
import { summarize } from "../lib/prom";
import { parseProm } from "../lib/prom";
import { getMockPromText } from "../lib/mock";

describe("MetaInspector", () => {
  const metrics = summarize(parseProm(getMockPromText()));

  it("renders the model + version pill", () => {
    render(
      <MetaInspector
        model="kairu-mock"
        version="0.7.0"
        metrics={metrics}
        generationFromMock={false}
        speculativeFromMock={true}
      />,
    );
    expect(screen.getByText("kairu-mock · v0.7.0")).toBeInTheDocument();
  });

  it("flips the generation pill when fromMock=true", () => {
    render(
      <MetaInspector
        model="m" metrics={metrics}
        generationFromMock={true} speculativeFromMock={true}
      />,
    );
    expect(screen.getByText("mock")).toBeInTheDocument();
  });

  it("shows live when generation is not mock", () => {
    render(
      <MetaInspector
        model="m" metrics={metrics}
        generationFromMock={false} speculativeFromMock={false}
      />,
    );
    expect(screen.getByText("live")).toBeInTheDocument();
  });
});
