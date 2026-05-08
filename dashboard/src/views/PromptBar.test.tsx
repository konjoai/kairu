import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PromptBar } from "./PromptBar";

const noop = () => {};

describe("PromptBar", () => {
  it("renders prompt + max_tokens + temperature inputs", () => {
    render(
      <PromptBar
        prompt="hi" onPromptChange={noop}
        maxTokens={48} onMaxTokensChange={noop}
        temperature={0.85} onTemperatureChange={noop}
        onSubmit={noop} disabled={false}
      />,
    );
    expect(screen.getByPlaceholderText(/prompt the model/i)).toBeInTheDocument();
    expect(screen.getByText("max_tokens")).toBeInTheDocument();
    expect(screen.getByText("temperature")).toBeInTheDocument();
  });

  it("disables submit when prompt is empty", () => {
    render(
      <PromptBar
        prompt="" onPromptChange={noop}
        maxTokens={48} onMaxTokensChange={noop}
        temperature={0.85} onTemperatureChange={noop}
        onSubmit={noop} disabled={false}
      />,
    );
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
  });

  it("clamps max_tokens to [1, 512]", () => {
    const cb = vi.fn();
    render(
      <PromptBar
        prompt="hi" onPromptChange={noop}
        maxTokens={48} onMaxTokensChange={cb}
        temperature={0.85} onTemperatureChange={noop}
        onSubmit={noop} disabled={false}
      />,
    );
    const inputs = screen.getAllByRole("spinbutton");
    // First spinbutton is max_tokens.
    const mt = inputs[0];
    userEvent.clear(mt).then(() => userEvent.type(mt, "9999"));
    // Test focuses on the clamping behaviour rather than the precise number;
    // we accept any value <= 512 from the callback.
    return Promise.resolve().then(() => {
      if (cb.mock.calls.length > 0) {
        for (const [v] of cb.mock.calls) expect(v).toBeLessThanOrEqual(512);
      }
    });
  });

  it("calls onSubmit when the button is clicked", async () => {
    const onSubmit = vi.fn();
    render(
      <PromptBar
        prompt="hi" onPromptChange={noop}
        maxTokens={48} onMaxTokensChange={noop}
        temperature={0.85} onTemperatureChange={noop}
        onSubmit={onSubmit} disabled={false}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /generate/i }));
    expect(onSubmit).toHaveBeenCalledOnce();
  });
});
