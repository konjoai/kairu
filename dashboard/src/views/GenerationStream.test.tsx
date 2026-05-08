import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GenerationStream } from "./GenerationStream";
import { buildMockGeneration } from "../lib/mock";

describe("GenerationStream", () => {
  it("renders the prompt block when prompt is supplied", () => {
    const g = buildMockGeneration();
    render(<GenerationStream tokens={g.tokens} streaming={false} prompt="hi" />);
    expect(screen.getByText("hi")).toBeInTheDocument();
  });

  it("shows the finish-reason summary when provided", () => {
    const g = buildMockGeneration();
    render(
      <GenerationStream
        tokens={g.tokens}
        streaming={false}
        finishReason="stop"
        totalSeconds={0.42}
      />,
    );
    expect(screen.getByText("stop")).toBeInTheDocument();
    expect(screen.getByText(/0\.42s/)).toBeInTheDocument();
  });

  it("renders nothing notable for empty tokens", () => {
    render(<GenerationStream tokens={[]} streaming />);
    expect(screen.getByText(/Generation/)).toBeInTheDocument();
  });
});
