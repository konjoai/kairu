/**
 * SSE / NDJSON parser for kairu's /generate endpoint.
 *
 * SSE frames:  `data: {json}\n\n`, terminated by `data: [DONE]\n\n`.
 * NDJSON:      `{json}\n` per line, no terminator.
 *
 * Auto-detects: if the buffer contains `\n\n`, treat as SSE (multi-line
 * blocks per frame). Otherwise NDJSON.
 */

export interface ParsedFrame {
  /** Raw JSON string after the `data:` prefix (SSE) or whole line (NDJSON). */
  json: string;
  /** True for the SSE [DONE] sentinel. */
  done: boolean;
}

export function parseStreamChunk(buffer: string): { frames: ParsedFrame[]; rest: string } {
  const frames: ParsedFrame[] = [];

  if (buffer.includes("\n\n")) {
    // SSE: split on \n\n, treat each part as a frame block.
    const parts = buffer.split("\n\n");
    const rest = parts.pop() ?? "";
    for (const block of parts) {
      for (const line of block.split("\n")) {
        if (!line || line.startsWith(":")) continue; // empty / comment
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") frames.push({ json: "", done: true });
          else if (payload) frames.push({ json: payload, done: false });
        }
      }
    }
    return { frames, rest };
  }

  // NDJSON: each line is a JSON object.
  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith(":")) continue;
    const payload = line.startsWith("data:") ? line.slice(5).trim() : line;
    if (payload === "[DONE]") frames.push({ json: "", done: true });
    else if (payload) frames.push({ json: payload, done: false });
  }
  return { frames, rest };
}
